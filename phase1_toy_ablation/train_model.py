#!/usr/bin/env python3
"""
Fully-trained causal transformer on synthetic (x, y, x', y') task.
Training only — saves checkpoints. Use analyze_model.py for analysis.
"""

from __future__ import annotations
import argparse, json, math, os, random, multiprocessing as mp
from dataclasses import dataclass, asdict
from typing import Dict
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F


# ───────────────────────── Config ───────────────────────── #

@dataclass
class Config:
    # data / task
    N: int = 128
    M: int | None = None
    p_train: float = 0.8
    p_eval:  float = 0.5
    use_bos: bool = False

    # model
    d_model: int = 64
    num_heads: int = 1
    num_layers: int = 3
    first_layer_mlp: bool = False
    dropout: float = 0.0
    use_rms: bool = False
    tie: bool = False
    normalize_embeddings: bool = False
    freeze_kv: bool = False

    # optimisation
    batch_size: int = 128
    lr: float = 2e-3
    weight_decay: float = 1e-2
    num_steps: int = 20_000

    # logging
    print_interval: int = 100
    save_interval: int = 2000
    early_save_interval: int = 0
    early_save_steps: int = 1000
    num_eval_samples: int = 2000
    num_eval_pairs: int = 50

    # misc
    seed: int = 2025
    num_workers: int = max(1, mp.cpu_count() - 1)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    output_path: str = "outputs"

    # feature flags
    freeze_embeddings: bool = False
    one_hot: bool = False
    no_norm: bool = False
    independent_truth: bool = False

    # derived (filled in finalise)
    vocab_size: int = 0
    seq_len: int = 0

    def finalise(self):
        if self.M is None:
            self.M = self.N
        self.vocab_size = 1 + self.N + self.M
        self.seq_len = 4 if self.use_bos else 3
        if self.one_hot:
            V = self.vocab_size
            L = self.seq_len
            target = 2 * V + L
            rem = target % self.num_heads
            if rem != 0:
                target += self.num_heads - rem
            self.d_model = target
        self.device = torch.device(self.device)


# ─────────────────────── Utilities ─────────────────────── #

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_random_mapping(cfg: Config) -> Dict[int, int]:
    rng = np.random.default_rng(cfg.seed)
    ins = np.arange(1, cfg.N + 1)
    outs = np.arange(cfg.N + 1, cfg.N + cfg.M + 1)
    return {int(i): int(o) for i, o in zip(ins, rng.permutation(outs))}


# ───────────────────── Data generation ───────────────────── #

def generate_single_sample(rng: np.random.Generator,
                           cfg: Config,
                           f_map: Dict[int, int],
                           *, is_train: bool):
    x, x_ = rng.integers(1, cfg.N + 1, 2)
    p = cfg.p_train if is_train else cfg.p_eval
    if cfg.independent_truth:
        t1 = rng.random() < p
        t2 = rng.random() < p
        y = f_map[x] if t1 else rng.integers(cfg.N + 1, cfg.N + cfg.M + 1)
        y_ = f_map[x_] if t2 else rng.integers(cfg.N + 1, cfg.N + cfg.M + 1)
        is_true = t1
    else:
        is_true = rng.random() < p
        if is_true:
            y, y_ = f_map[x], f_map[x_]
        else:
            y = rng.integers(cfg.N + 1, cfg.N + cfg.M + 1)
            y_ = rng.integers(cfg.N + 1, cfg.N + cfg.M + 1)
    seq = [0, x, y, x_, y_] if cfg.use_bos else [x, y, x_, y_]
    return seq, is_true


def gen_batch(rng: np.random.Generator,
              batch_size: int,
              cfg: Config,
              f_map: Dict[int, int],
              is_train: bool = True):
    seqs, labs = [], []
    for _ in range(batch_size):
        s, l = generate_single_sample(rng, cfg, f_map, is_train=is_train)
        seqs += s
        labs.append(1 if l else 0)
    full = np.asarray(seqs, dtype=np.int64).reshape(batch_size, cfg.seq_len + 1)
    return full, np.asarray(labs, dtype=np.int64)


def _batch_worker(q, batch_size, cfg_dict, f_map, is_train, seed_vec):
    cfg = Config(**{k: v for k, v in cfg_dict.items()
                    if k in Config.__dataclass_fields__})
    cfg.finalise()
    rng = np.random.default_rng(seed_vec)
    while True:
        full, lab = gen_batch(rng, batch_size, cfg, f_map, is_train)
        q.put((full, lab))


def iterate_batches(cfg: Config,
                    f_map: Dict[int, int],
                    seed: int = 42,
                    is_train: bool = True):
    cfg_dict = asdict(cfg)
    cfg_dict['device'] = str(cfg.device)

    q = mp.Queue(maxsize=10000)
    procs = []
    for i in range(cfg.num_workers):
        p = mp.Process(target=_batch_worker,
                       args=(q, cfg.batch_size, cfg_dict, f_map, is_train, [seed, i]),
                       daemon=True)
        p.start()
        procs.append(p)

    try:
        while True:
            full, lab = q.get()
            yield full[:, :-1], full[:, 1:], lab
    finally:
        q.close()
        for p in procs:
            p.terminate()
            p.join()


# ───────────────────── Model definition ───────────────────── #

class CausalSelfAttention(nn.Module):
    def __init__(self, d: int, h: int):
        super().__init__()
        assert d % h == 0
        self.h, self.dh = h, d // h
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.bias = nn.Parameter(torch.zeros(1, 1, d))

    def forward(self, x):
        B, T, D = x.shape
        q = self.q(x).view(B, T, self.h, self.dh).transpose(1, 2)
        k = self.k(x).view(B, T, self.h, self.dh).transpose(1, 2)
        v = self.v(x).view(B, T, self.h, self.dh).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)
        scores.masked_fill_(
            torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), 1),
            float('-inf'))
        attn = F.softmax(scores, dim=-1)
        self.attn_weights = attn.detach()
        y = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.o(y)


class TransformerBlock(nn.Module):
    def __init__(self, d: int, h: int, *, no_attn=False, no_mlp=False,
                 no_norm=False, dropout=0., use_rms=False):
        super().__init__()
        self.attn = None if no_attn else CausalSelfAttention(d, h)
        self.mlp = None if no_mlp else nn.Sequential(
            nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

        NormClass = nn.RMSNorm if use_rms else nn.LayerNorm
        self.n1 = NormClass(d, elementwise_affine=False) if not no_norm else nn.Identity()
        self.n2 = NormClass(d, elementwise_affine=False) if not no_norm else nn.Identity()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        if self.attn is not None:
            x = self.n1(x + self.drop(self.attn(x)))
        if self.mlp is not None:
            x = self.n2(x + self.drop(self.mlp(x)))
        return x


class CausalTransformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Parameter(torch.zeros(1, cfg.seq_len, cfg.d_model))
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg.d_model, cfg.num_heads,
                             no_attn=(cfg.first_layer_mlp and i == 0),
                             no_mlp=(not cfg.first_layer_mlp or i > 0),
                             no_norm=cfg.no_norm, use_rms=cfg.use_rms)
            for i in range(cfg.num_layers)])
        self.out = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.apply(lambda m: nn.init.normal_(m.weight, 0., 0.02)
                   if isinstance(m, (nn.Linear, nn.Embedding)) else None)

    def forward(self, x, *, return_activations=False):
        h = self.tok(x) + self.pos[:, :x.size(1)]
        acts = [h]
        for blk in self.blocks:
            h = blk(h)
            acts.append(h)
        logits = self.out(h)
        return (logits, acts) if return_activations else logits


# ───────────────────── Checkpoint ───────────────────── #

def save_checkpoint(model, f_map, cfg, step):
    os.makedirs(os.path.join(cfg.output_path, "checkpoints"), exist_ok=True)
    cfg_dict = asdict(cfg)
    cfg_dict['device'] = str(cfg.device)
    torch.save({
        'step': step,
        'state': {k: v.cpu() for k, v in model.state_dict().items()},
        'f_map': f_map,
        'cfg': cfg_dict,
    }, os.path.join(cfg.output_path, f"checkpoints/ckpt_step{step}.pt"))


# ──────────────────── Training loop ──────────────────── #

def train(cfg: Config):
    seed_everything(cfg.seed)
    f_map = create_random_mapping(cfg)
    model = CausalTransformer(cfg).to(cfg.device)

    if cfg.one_hot:
        V, L = cfg.vocab_size, cfg.seq_len
        model.tok.weight.data.zero_()
        model.tok.weight.data[torch.arange(V), torch.arange(V)] = 1
        model.pos.data.zero_()
        model.pos.data[0, torch.arange(L), 2 * V + torch.arange(L)] = 1
        model.out.weight.data.zero_()
        model.out.weight.data[torch.arange(V), V + torch.arange(V)] = 1

    if cfg.freeze_embeddings:
        model.tok.requires_grad_(False)
        model.pos.requires_grad_(False)
        model.out.requires_grad_(False)

    if cfg.freeze_kv:
        model.blocks[0].attn.q.weight.data.zero_()
        model.blocks[0].attn.q.weight.requires_grad_(False)
        model.blocks[0].attn.k.weight.data.zero_()
        model.blocks[0].attn.k.weight.requires_grad_(False)

    if cfg.normalize_embeddings:
        with torch.no_grad():
            model.tok.weight /= model.tok.weight.norm(dim=-1, keepdim=True)
            model.out.weight /= model.out.weight.norm(dim=-1, keepdim=True)

    if cfg.tie:
        model.tok.weight = model.out.weight

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    crit = nn.CrossEntropyLoss(reduction='none')

    os.makedirs(cfg.output_path, exist_ok=True)
    with open(os.path.join(cfg.output_path, "config.json"), "w") as f:
        cfg_dict = asdict(cfg)
        cfg_dict['device'] = str(cfg.device)
        json.dump(cfg_dict, f, indent=2)

    save_checkpoint(model, f_map, cfg, 0)

    batches = iterate_batches(cfg, f_map, is_train=True)
    model.train()

    for step in range(cfg.num_steps):
        inp, tgt, lab = next(batches)
        inp = torch.as_tensor(inp, device=cfg.device)
        tgt = torch.as_tensor(tgt, device=cfg.device)

        opt.zero_grad()
        logits = model(inp).view(-1, cfg.vocab_size)
        loss = crit(logits, tgt.view(-1)).mean()
        loss.backward()
        opt.step()

        if cfg.normalize_embeddings:
            with torch.no_grad():
                model.tok.weight /= model.tok.weight.norm(dim=-1, keepdim=True)
                model.out.weight /= model.out.weight.norm(dim=-1, keepdim=True)

        if step % cfg.print_interval == 0:
            bs = inp.size(0)
            bos = 1 if cfg.use_bos else 0
            with torch.no_grad():
                lt = crit(logits, tgt.view(-1)).view(bs, -1)
                mask = (lab == 1)
                if mask.any():
                    l1 = lt[mask, bos].mean().item()
                    l2 = lt[mask, bos + 2].mean().item()
                    probs = torch.softmax(logits.view(bs, -1, cfg.vocab_size), dim=-1)
                    tgt_r = tgt.view(bs, -1)
                    p1 = probs[mask, bos, tgt_r[mask, bos]].mean().item()
                    p2 = probs[mask, bos + 2, tgt_r[mask, bos + 2]].mean().item()
                else:
                    l1 = l2 = p1 = p2 = float('nan')
            print(f"Step {step}/{cfg.num_steps}  loss={loss.item():.4f}  "
                  f"y_loss={l1:.4f}  y'_loss={l2:.4f}  "
                  f"y_prob={p1:.4f}  y'_prob={p2:.4f}")

        s = step + 1
        interval = cfg.early_save_interval if (cfg.early_save_interval and s <= cfg.early_save_steps) else cfg.save_interval
        if s % interval == 0:
            save_checkpoint(model, f_map, cfg, s)

    save_checkpoint(model, f_map, cfg, cfg.num_steps)
    print(f"Training complete. {cfg.num_steps} steps. "
          f"Checkpoints in {cfg.output_path}/checkpoints/")


# ───────────────────────── CLI ───────────────────────── #

def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--N', type=int, default=64)
    p.add_argument('--M', type=int, default=None)
    p.add_argument('--p_train', type=float, default=0.98)
    p.add_argument('--p_eval', type=float, default=0.5)
    p.add_argument('--d_model', type=int, default=128)
    p.add_argument('--num_heads', type=int, default=1)
    p.add_argument('--num_layers', type=int, default=1)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=1e-3)
    p.add_argument('--first_layer_mlp', action='store_true')
    p.add_argument('--num_steps', type=int, default=20000)
    p.add_argument('--print_interval', type=int, default=100)
    p.add_argument('--save_interval', type=int, default=2000)
    p.add_argument('--early_save_interval', type=int, default=0,
                   help="Denser save interval for early training (0=disabled)")
    p.add_argument('--early_save_steps', type=int, default=1000,
                   help="Use early_save_interval for the first N steps")
    p.add_argument('--freeze_embeddings', action='store_true')
    p.add_argument('--one_hot', action='store_true')
    p.add_argument('--no_norm', action='store_true')
    p.add_argument('--use_rms', action='store_true')
    p.add_argument('--use_bos', action='store_true')
    p.add_argument('--normalize_embeddings', action='store_true')
    p.add_argument('--tie', action='store_true')
    p.add_argument('--freeze_kv', action='store_true')
    p.add_argument('--independent_truth', action='store_true')
    p.add_argument('--output_path', type=str, default='outputs')
    p.add_argument('--seed', type=int, default=1)
    return p


if __name__ == "__main__":
    cfg = Config(**vars(build_parser().parse_args()))
    cfg.finalise()
    train(cfg)
