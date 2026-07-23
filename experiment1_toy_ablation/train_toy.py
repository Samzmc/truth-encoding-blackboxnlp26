import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import argparse
import os
import json
import multiprocessing as mp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters
N = 20
M = N
VOCAB_SIZE = N + M
P_CORRECT_TRAIN = 0.8
P_CORRECT_EVAL = 0.5
SEQ_LENGTH = 3  # input is [x, y, x'], full sequence is [x, y, x', y']
EMBEDDING_DIM = 2 * VOCAB_SIZE + SEQ_LENGTH
NUM_HEADS = 1
BATCH_SIZE = 16
NUM_LAYERS = 1
LEARNING_RATE = 1
USE_FIRST_LAYER_MLP = False

# Token mappings
INPUT_TOKENS = list(range(0, N))
OUTPUT_TOKENS = list(range(N, N + M))


# ───────────────────── Data generation ───────────────────── #

def create_random_mapping(seed=None):
    rng = np.random.default_rng(seed)
    output_indices = rng.permutation(N)
    return {INPUT_TOKENS[i]: OUTPUT_TOKENS[output_indices[i]] for i in range(N)}


def generate_single_sample(rng, f_map, is_train=True, independent_truth=False):
    x = rng.choice(INPUT_TOKENS)
    x_prime = rng.choice(INPUT_TOKENS)

    p = P_CORRECT_TRAIN if is_train else P_CORRECT_EVAL
    if independent_truth:
        t1 = rng.random() < p
        t2 = rng.random() < p
        y = f_map[x] if t1 else rng.choice(OUTPUT_TOKENS)
        y_prime = f_map[x_prime] if t2 else rng.choice(OUTPUT_TOKENS)
        is_true = t1
    else:
        is_true = rng.random() < p
        if is_true:
            y = f_map[x]
            y_prime = f_map[x_prime]
        else:
            y = rng.choice(OUTPUT_TOKENS)
            y_prime = rng.choice(OUTPUT_TOKENS)

    return [x, y, x_prime, y_prime], is_true


def gen_batch(rng, batch_size, f_map, is_train=True, independent_truth=False):
    seqs = []
    is_trues = []
    for _ in range(batch_size):
        seq, is_true = generate_single_sample(rng, f_map, is_train,
                                              independent_truth=independent_truth)
        seqs += seq
        is_trues.append(1 if is_true else 0)
    x = np.array(seqs, dtype=np.int64).reshape(batch_size, SEQ_LENGTH + 1)
    return x, np.array(is_trues, dtype=np.int64)


def _batch_worker(queue, batch_size, f_map, is_train, seed_vec, independent_truth):
    rng = np.random.default_rng(seed_vec)
    while True:
        x, is_true = gen_batch(rng, batch_size, f_map, is_train,
                               independent_truth=independent_truth)
        queue.put((x, is_true))


def iterate_batches(f_map, batch_size=20, seed=42, is_train=True, independent_truth=False):
    num_cores = max(1, mp.cpu_count() - 1)
    q = mp.Queue(maxsize=10000)
    procs = [mp.Process(target=_batch_worker,
                        args=(q, batch_size, f_map, is_train, [seed, i],
                              independent_truth),
                        daemon=True)
             for i in range(num_cores)]
    for p in procs:
        p.start()

    try:
        while True:
            x, is_true = q.get()
            yield x[:, :-1], x[:, 1:], is_true
    finally:
        for p in procs:
            p.terminate()
            p.join()


# ───────────────────── Model ───────────────────── #

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, D = x.size()
        q = self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        scores.masked_fill_(mask, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        self.attn_weights = attn_weights.detach()
        out = (attn_weights @ v).permute(0, 2, 1, 3).reshape(B, T, D)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, no_attention=False, no_mlp=False, no_norm=False, dropout=0):
        super().__init__()
        self.attention = CausalSelfAttention(d_model, num_heads) if not no_attention else None
        self.norm1 = nn.RMSNorm(d_model, elementwise_affine=False) if not no_norm else nn.Identity()
        self.norm2 = nn.RMSNorm(d_model, elementwise_affine=False) if not no_norm else nn.Identity()
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        ) if not no_mlp else None
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if self.attention is not None:
            x = self.norm1(x + self.dropout(self.attention(x)))
        if self.feed_forward is not None:
            x = self.norm2(x + self.dropout(self.feed_forward(x)))
        return x


class CausalTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers,
                 dropout=0, first_layer_mlp=False, no_norm=False):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_encoding = nn.Parameter(torch.zeros(1, SEQ_LENGTH, d_model))

        if first_layer_mlp:
            self.transformer_blocks = nn.ModuleList([
                TransformerBlock(d_model, num_heads,
                                no_attention=(i == 0), no_mlp=(i > 0),
                                no_norm=no_norm, dropout=dropout)
                for i in range(num_layers)
            ])
        else:
            self.transformer_blocks = nn.ModuleList([
                TransformerBlock(d_model, num_heads,
                                no_attention=False, no_mlp=True,
                                no_norm=no_norm, dropout=dropout)
                for i in range(num_layers)
            ])

        self.output_layer = nn.Linear(d_model, vocab_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, return_activations=False):
        tok = self.token_embedding(x)
        h = self.dropout(tok + self.position_encoding[:, :x.size(1), :])

        activations = [h]
        for block in self.transformer_blocks:
            h = block(h)
            activations.append(h)

        logits = self.output_layer(h)
        return (logits, activations) if return_activations else logits


# ───────────────────── Checkpoint helpers ───────────────────── #

def save_checkpoint(model, f_map, config, step, output_path):
    os.makedirs(os.path.join(output_path, "checkpoints"), exist_ok=True)
    torch.save({
        'step': step,
        'state': {k: v.cpu() for k, v in model.state_dict().items()},
        'f_map': f_map,
        'config': config,
    }, os.path.join(output_path, f"checkpoints/ckpt_step{step}.pt"))


# ───────────────────── Training ───────────────────── #

def train_model(num_steps=1000, freeze_embeddings=False, one_hot=False,
                no_norm=False, seed=42, output_path="outputs",
                save_interval=100, print_interval=100,
                independent_truth=False):

    f_map = create_random_mapping(seed=seed)

    config = {
        'N': N, 'M': M, 'vocab_size': VOCAB_SIZE,
        'seq_length': SEQ_LENGTH, 'd_model': EMBEDDING_DIM,
        'num_heads': NUM_HEADS, 'num_layers': NUM_LAYERS,
        'batch_size': BATCH_SIZE, 'lr': LEARNING_RATE,
        'p_train': P_CORRECT_TRAIN, 'p_eval': P_CORRECT_EVAL,
        'one_hot': one_hot, 'freeze_embeddings': freeze_embeddings,
        'no_norm': no_norm, 'first_layer_mlp': USE_FIRST_LAYER_MLP,
        'independent_truth': independent_truth, 'seed': seed,
    }

    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    model = CausalTransformer(
        vocab_size=VOCAB_SIZE, d_model=EMBEDDING_DIM,
        num_heads=NUM_HEADS, num_layers=NUM_LAYERS,
        first_layer_mlp=USE_FIRST_LAYER_MLP, no_norm=no_norm,
    ).to(device)

    # ── one-hot initialisation ──
    if one_hot:
        V, L = VOCAB_SIZE, SEQ_LENGTH
        assert EMBEDDING_DIM == 2 * V + L

        model.token_embedding.weight.data.zero_()
        model.token_embedding.weight.data[torch.arange(V), torch.arange(V)] = 1

        model.position_encoding.data.zero_()
        model.position_encoding.data[0, torch.arange(L), V + torch.arange(L)] = 1

        model.output_layer.weight.data.zero_()
        model.output_layer.weight.data[torch.arange(V), V + L + torch.arange(V)] = 1

    # ── freeze embedding/unembedding ──
    if freeze_embeddings:
        model.token_embedding.requires_grad_(False)
        model.position_encoding.requires_grad_(False)
        model.output_layer.requires_grad_(False)

    # ── uniform attention: freeze Q and K to zero ──
    for block in model.transformer_blocks:
        if block.attention is not None:
            block.attention.q_proj.weight.data.zero_()
            block.attention.q_proj.weight.requires_grad_(False)
            block.attention.k_proj.weight.data.zero_()
            block.attention.k_proj.weight.requires_grad_(False)

    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    save_checkpoint(model, f_map, config, 0, output_path)

    model.train()
    step = 0
    for inputs, targets, is_true in iterate_batches(f_map, batch_size=BATCH_SIZE,
                                                     seed=seed, is_train=True,
                                                     independent_truth=independent_truth):
        if step >= num_steps:
            break

        inputs = torch.from_numpy(inputs).to(device)
        targets = torch.from_numpy(targets).to(device)
        is_true = torch.from_numpy(is_true).to(device)

        logits = model(inputs)
        logits_flat = logits.view(-1, VOCAB_SIZE)
        targets_flat = targets.view(-1)

        all_loss = criterion(logits_flat, targets_flat)
        loss = all_loss.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ── monitoring (no grad) ──
        if (step + 1) % print_interval == 0:
            bs = inputs.size(0)
            with torch.no_grad():
                per_pos = all_loss.view(bs, -1)
                true_mask = (is_true == 1)
                if true_mask.any():
                    l1 = per_pos[true_mask, 0].mean().item()
                    l2 = per_pos[true_mask, 2].mean().item()
                    probs = torch.softmax(logits, dim=-1)
                    p1 = probs[true_mask, 0, targets[true_mask, 0]].mean().item()
                    p2 = probs[true_mask, 2, targets[true_mask, 2]].mean().item()
                else:
                    l1 = l2 = p1 = p2 = float('nan')
            print(f"Step {step+1}/{num_steps}  loss={loss.item():.4f}  "
                  f"y_loss={l1:.4f}  y'_loss={l2:.4f}  "
                  f"y_prob={p1:.4f}  y'_prob={p2:.4f}")

        # ── periodic checkpoint ──
        if (step + 1) % save_interval == 0:
            save_checkpoint(model, f_map, config, step + 1, output_path)

        step += 1

    # ── final checkpoint ──
    save_checkpoint(model, f_map, config, step, output_path)
    print(f"Training complete. {step} steps. Checkpoints in {output_path}/checkpoints/")
    return model, f_map


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--one_hot", action="store_true")
    parser.add_argument("--freeze_embeddings", action="store_true")
    parser.add_argument("--no_norm", action="store_true")
    parser.add_argument("--output_path", type=str, default="outputs")
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--print_interval", type=int, default=100)
    parser.add_argument("--independent_truth", action="store_true")
    args = parser.parse_args()

    train_model(
        num_steps=args.num_steps,
        seed=args.seed,
        one_hot=args.one_hot,
        freeze_embeddings=args.freeze_embeddings,
        no_norm=args.no_norm,
        output_path=args.output_path,
        save_interval=args.save_interval,
        print_interval=args.print_interval,
        independent_truth=args.independent_truth,
    )
