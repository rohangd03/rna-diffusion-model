# diffusion.py
import math
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import RNA
from eval_harness import evaluate
from baseline import make_targets

NUC = "ACGU"
MASK = 4          # sequence token: masked position
PAD_SEQ = 5       # sequence token: padding
STRUCT = {".": 0, "(": 1, ")": 2}
PAD_S = 3         # structure token: padding
NULL_S = 4        # structure token: "no condition" (for classifier-free guidance)
MAX_LEN = 24
STEPS = 12        # number of unmasking steps at sampling time

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


# ---------- data ----------
def make_training_data(n=20000, seed=0):
    """Random sequences folded with ViennaRNA. Keep (seq, structure) pairs with at least one pair."""
    rng = random.Random(seed)
    pairs = []
    while len(pairs) < n:
        L = rng.randint(12, MAX_LEN)
        seq = "".join(rng.choice(NUC) for _ in range(L))
        struct, _ = RNA.fold(seq)
        if "(" in struct:
            pairs.append((seq, struct))
    return pairs


def to_tensors(pairs):
    S = torch.full((len(pairs), MAX_LEN), PAD_SEQ, dtype=torch.long)
    X = torch.full((len(pairs), MAX_LEN), PAD_S, dtype=torch.long)
    for i, (s, x) in enumerate(pairs):
        L = len(s)
        S[i, :L] = torch.tensor([NUC.index(c) for c in s])
        X[i, :L] = torch.tensor([STRUCT[c] for c in x])
    return S.to(DEVICE), X.to(DEVICE)


# ---------- model ----------
class Denoiser(nn.Module):
    def __init__(self, d=128, layers=4, heads=4):
        super().__init__()
        self.seq_emb = nn.Embedding(6, d)
        self.str_emb = nn.Embedding(5, d)
        self.pos_emb = nn.Embedding(MAX_LEN, d)
        self.t_mlp = nn.Sequential(nn.Linear(1, d), nn.SiLU(), nn.Linear(d, d))
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.out = nn.Linear(d, 4)

    def forward(self, seq, struct, t, pad_mask):
        pos = torch.arange(seq.size(1), device=seq.device)
        h = (self.seq_emb(seq) + self.str_emb(struct)
             + self.pos_emb(pos)[None] + self.t_mlp(t[:, None])[:, None])
        h = self.enc(h, src_key_padding_mask=pad_mask)
        return self.out(h)  # (B, L, 4) logits over A, C, G, U


# ---------- training ----------
def train(model, S, X, steps=10000, bs=128, lr=3e-4):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    N = S.size(0)
    for step in range(steps):
        idx = torch.randint(0, N, (bs,), device=DEVICE)
        seq, struct = S[idx], X[idx].clone()
        valid = seq != PAD_SEQ
        pad = ~valid

        # forward (corruption): mask each nucleotide with prob t ~ U(0,1)
        t = torch.rand(bs, device=DEVICE)
        m = (torch.rand(seq.shape, device=DEVICE) < t[:, None]) & valid
        noisy = seq.clone()
        noisy[m] = MASK

        # classifier-free guidance: drop the structure condition 10% of the time
        drop = torch.rand(bs, device=DEVICE) < 0.1
        struct[drop] = NULL_S

        logits = model(noisy, struct, t, pad)
        loss = F.cross_entropy(logits[m], seq[m]) if m.any() else logits.sum() * 0

        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 500 == 0:
            print(f"step {step:5d}  loss {loss.item():.4f}")


# ---------- sampling ----------
@torch.no_grad()
def design(model, target, n=1, guidance=2.0, temp=1.0):
    """Design n sequences for a dot-bracket target. Returns list of strings."""
    L = len(target)
    x = torch.full((n, MAX_LEN), PAD_S, dtype=torch.long, device=DEVICE)
    x[:, :L] = torch.tensor([STRUCT[c] for c in target], device=DEVICE)
    x_null = torch.full((n, MAX_LEN), PAD_S, dtype=torch.long, device=DEVICE)
    x_null[:, :L] = NULL_S
    seq = torch.full((n, MAX_LEN), PAD_SEQ, dtype=torch.long, device=DEVICE)
    seq[:, :L] = MASK
    pad = (torch.arange(MAX_LEN, device=DEVICE) >= L)[None].expand(n, -1)
    k = math.ceil(L / STEPS)

    for step in range(STEPS):
        t = torch.full((n,), 1 - step / STEPS, device=DEVICE)
        masked = seq == MASK
        lc = model(seq, x, t, pad)
        lu = model(seq, x_null, t, pad)
        logits = lu + guidance * (lc - lu)           # classifier-free guidance
        probs = F.softmax(logits / temp, dim=-1)
        conf, pred = probs.max(dim=-1)

        # reveal the k most confident masked positions this step
        conf = conf.masked_fill(~masked, -1.0)
        idx = conf.topk(min(k, L), dim=1).indices
        chosen = torch.zeros_like(masked).scatter_(1, idx, True)
        seq = torch.where(chosen & masked, pred, seq)

    return ["".join(NUC[i] for i in row[:L].tolist()) for row in seq.cpu()]


# ---------- main ----------
if __name__ == "__main__":
    torch.manual_seed(0)
    random.seed(0)

    print("Building training data...")
    pairs = make_training_data(n=20000, seed=0)
    S, X = to_tensors(pairs)

    print("Training...")
    model = Denoiser().to(DEVICE)
    train(model, S, X, steps=10000)
    model.eval()

    print("Designing benchmark targets...")
    targets = make_targets(n=30, seed=0)   # same benchmark as baseline.py
    designs = [design(model, t, n=1)[0] for t in targets]

    res = evaluate(designs, targets)
    print(f"\nDiffusion model: n={res['n']}  success={res['success_rate']:.2f}  "
          f"mean ensemble defect={res['mean_ensemble_defect']:.3f}")

    with open("baseline_results.json") as f:
        base = json.load(f)
    print(f"RNAinverse:      n={base['n']}  success={base['success_rate']:.2f}  "
          f"mean ensemble defect={base['mean_ensemble_defect']:.3f}")

    with open("diffusion_results.json", "w") as f:
        json.dump(res, f, indent=2, default=str)
    print("Saved diffusion_results.json")