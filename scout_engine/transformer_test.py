"""
transformer_test.py

Tests whether a Transformer can learn to compute Player._hand_value() for
random subsets of card sequences sampled from Scout hands.

Usage:
    python -m scout_engine.transformer_test
    python -m scout_engine.transformer_test --lr 1e-3 --embed_dim 64 --num_samples 20000
"""

import argparse
import os
import random
import sys
import warnings

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from .common import Player, Util

# ─── Label function ──────────────────────────────────────────────────────────

def hand_value(values: list[int]) -> float:
    """Heuristic value for a card sequence — delegates to Player._hand_value."""
    return Player._hand_value(None, values)  # type: ignore[arg-type]  # _hand_value doesn't use self


# ─── Dataset ─────────────────────────────────────────────────────────────────

_HAND_SIZE = 9  # cards per player in a 5-player game


def generate_samples(num_samples: int) -> list[tuple[list[int], float]]:
    """
    Generate (card_sequence, hand_value) pairs by sampling random card values
    (1–10, with replacement) at the same hand size as a 5-player game.
    """
    data: list[tuple[list[int], float]] = []
    for _ in range(num_samples):
        card_values = [random.randint(1, 10) for _ in range(_HAND_SIZE)]
        length = random.randint(1, _HAND_SIZE)
        indices = sorted(random.sample(range(_HAND_SIZE), length))
        subset = [card_values[i] for i in indices]
        data.append((subset, hand_value(subset)))
    return data


# Token IDs: 0 = [CLS], 1–10 = card values, -1 used as pad sentinel in tensors
CLS_TOKEN = 0
PAD_SENTINEL = -1


class CardSubsetDataset(Dataset):
    def __init__(self, samples: list[tuple[list[int], float]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        cards, value = self.samples[idx]
        tokens = torch.tensor([CLS_TOKEN] + cards, dtype=torch.long)
        return tokens, torch.tensor(value, dtype=torch.float32)


def collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor]]):
    """Pad token sequences with PAD_SENTINEL (-1) to the max length in the batch."""
    seqs, values = zip(*batch)
    padded = torch.nn.utils.rnn.pad_sequence(
        seqs, batch_first=True, padding_value=PAD_SENTINEL)
    return padded, torch.stack(values)


# ─── Model ───────────────────────────────────────────────────────────────────

class HandValueTransformer(nn.Module):
    """
    TransformerEncoder with a [CLS] token. Each card is embedded via a
    learnable token embedding (card values 1–10) plus a learnable position
    embedding. The [CLS] output is fed through a 128→64→1 MLP.
    """
    VOCAB_SIZE = 11  # 0=[CLS], 1-10=card values

    def __init__(self, embed_dim: int, num_heads: int, num_layers: int,
                 dim_ffd: int, dropout: float = 0.0, max_seq_len: int = 12):
        super().__init__()
        self.token_emb = nn.Embedding(self.VOCAB_SIZE, embed_dim)
        self.pos_emb = nn.Embedding(max_seq_len + 1, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=dim_ffd,
            batch_first=True, norm_first=True, dropout=dropout)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: [B, L] long — PAD_SENTINEL (-1) marks padding.
        returns: [B] predicted hand values.
        """
        pad_mask = tokens == PAD_SENTINEL          # [B, L] True = ignore
        safe = tokens.clamp(min=0)                 # replace -1 with 0 for embedding
        B, L = safe.shape
        positions = torch.arange(L, device=tokens.device).unsqueeze(0).expand(B, -1)
        embedded = self.token_emb(safe) + self.pos_emb(positions)   # [B, L, E]
        transformed = self.transformer(
            embedded, src_key_padding_mask=pad_mask)                 # [B, L, E]
        cls_out = transformed[:, 0, :]                               # [B, E]
        return self.mlp(cls_out).squeeze(1)                          # [B]


# ─── Training ────────────────────────────────────────────────────────────────

def evaluate(model: nn.Module, loader: DataLoader,
             loss_fn: nn.Module, device: torch.device) -> tuple[float, float]:
    """Returns (mean_mse_loss, accuracy) where accuracy = fraction with |pred−true| < 0.5."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for tokens, values in loader:
            tokens, values = tokens.to(device), values.to(device)
            preds = model(tokens)
            total_loss += loss_fn(preds, values).item() * len(values)
            correct += (preds - values).abs().lt(0.5).sum().item()
            total += len(values)
    return total_loss / total, correct / total


def _plot(model: nn.Module, test_loader: DataLoader, device: torch.device,
          epochs: list[int], train_loss: list[float],
          val_loss: list[float], val_acc: list[float]) -> None:
    """Render training curves, error histogram, and true-vs-predicted scatter."""
    try:
        import matplotlib
        has_display = _display_available()
        matplotlib.use("Agg" if not has_display else matplotlib.get_backend())
        import matplotlib.pyplot as plt
        import numpy as np
        from scipy.stats import norm as _norm
    except ImportError as e:
        print(f"{e} — skipping plots.")
        return

    # Collect test predictions
    model.eval()
    all_true: list[float] = []
    all_pred: list[float] = []
    with torch.no_grad():
        for tokens, values in test_loader:
            tokens, values = tokens.to(device), values.to(device)
            preds = model(tokens)
            all_true.extend(values.cpu().tolist())
            all_pred.extend(preds.cpu().tolist())
    diffs = [t - p for t, p in zip(all_true, all_pred)]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Transformer hand-value prediction")
    (ax_loss, ax_acc), (ax_hist, ax_scatter) = axes

    # 1. Loss curves
    ax_loss.plot(epochs, train_loss, label="train loss")
    ax_loss.plot(epochs, val_loss, label="val loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("MSE loss")
    ax_loss.set_title("Loss over training")
    ax_loss.legend()

    # 2. Validation accuracy curve
    ax_acc.plot(epochs, val_acc, color="tab:green", label="val acc")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy (|err| < 0.5)")
    ax_acc.set_title("Validation accuracy over training")
    ax_acc.set_ylim(0, 1)
    ax_acc.legend()

    # 3. Histogram of score differences (true − predicted) + fitted normal
    diffs_arr = np.array(diffs)
    mu, sigma = diffs_arr.mean(), diffs_arr.std()
    n_bins = 40
    counts, bin_edges, _ = ax_hist.hist(
        diffs_arr, bins=n_bins, color="tab:blue", alpha=0.6, label="true − pred")
    x_fit = np.linspace(bin_edges[0], bin_edges[-1], 300)
    bin_width = bin_edges[1] - bin_edges[0]
    ax_hist.plot(x_fit, _norm.pdf(x_fit, mu, sigma) * len(diffs) * bin_width,
                 color="tab:red", linewidth=1.5,
                 label=f"normal fit  μ={mu:.2f}, σ={sigma:.2f}")
    ax_hist.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax_hist.set_xlabel("true − predicted")
    ax_hist.set_ylabel("Count")
    ax_hist.set_title("Error distribution (test set)")
    ax_hist.legend()

    # 4. True vs predicted scatter
    ax_scatter.scatter(all_true, all_pred, alpha=0.3, s=8, color="tab:orange")
    lo = min(min(all_true), min(all_pred))
    hi = max(max(all_true), max(all_pred))
    ax_scatter.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, label="y = x")
    ax_scatter.set_xlabel("True value")
    ax_scatter.set_ylabel("Predicted value")
    ax_scatter.set_title("True vs predicted (test set)")
    ax_scatter.legend()

    fig.tight_layout()

    if has_display:
        plt.show()
    else:
        fig.savefig("transformer_test_plots.png", dpi=150)
        print("Plots saved to transformer_test_plots.png")
    plt.close("all")


def _display_available() -> bool:
    """Return True if a graphical display is likely available."""
    if sys.platform == "darwin":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def train(args: argparse.Namespace) -> None:
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    print(f"Generating {args.num_samples:,} samples...")
    samples = generate_samples(args.num_samples)
    dataset = CardSubsetDataset(samples)

    n = len(dataset)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    n_test = n - n_train - n_val
    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test])
    print(f"Split: {n_train} train / {n_val} val / {n_test} test")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HandValueTransformer(
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dim_ffd=args.dim_ffd,
        dropout=args.dropout,
        max_seq_len=12,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Device: {device}  |  Parameters: {num_params:,}")
    print(f"Embed dim: {args.embed_dim}  heads: {args.num_heads}  "
          f"layers: {args.num_layers}  ffd: {args.dim_ffd}  "
          f"dropout: {args.dropout}  lr: {args.lr}")
    print()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    history_epochs: list[int] = []
    history_train_loss: list[float] = []
    history_val_loss: list[float] = []
    history_val_acc: list[float] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        n_seen = 0
        for tokens, values in train_loader:
            tokens, values = tokens.to(device), values.to(device)
            preds = model(tokens)
            loss = loss_fn(preds, values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(values)
            n_seen += len(values)
        train_loss /= n_seen

        if epoch % args.log_interval == 0 or epoch == 1:
            val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
            history_epochs.append(epoch)
            history_train_loss.append(train_loss)
            history_val_loss.append(val_loss)
            history_val_acc.append(val_acc)
            print(f"Epoch {epoch:4d}/{args.epochs}  "
                  f"train_loss={train_loss:.4f}  "
                  f"val_loss={val_loss:.4f}  "
                  f"val_acc={val_acc:.3f}")

    test_loss, test_acc = evaluate(model, test_loader, loss_fn, device)
    print(f"\nTest  loss={test_loss:.4f}  acc={test_acc:.3f}  "
          f"(acc = fraction with |pred−true| < 0.5)")

    if args.plot:
        _plot(model, test_loader, device,
              history_epochs, history_train_loss, history_val_loss, history_val_acc)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a Transformer to predict Scout hand values.")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Adam learning rate (default: 1e-3)")
    parser.add_argument("--embed_dim", type=int, default=64,
                        help="Token embedding dimension (default: 64)")
    parser.add_argument("--num_heads", type=int, default=4,
                        help="Number of attention heads (default: 4)")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="Number of TransformerEncoder layers (default: 2)")
    parser.add_argument("--dim_ffd", type=int, default=128,
                        help="Feedforward dimension inside transformer (default: 128)")
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="Dropout rate in transformer encoder layers (default: 0.0)")
    parser.add_argument("--num_samples", type=int, default=10000,
                        help="Total dataset size (default: 10000)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Training batch size (default: 64)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs (default: 50)")
    parser.add_argument("--log_interval", type=int, default=5,
                        help="Print metrics every N epochs (default: 5)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (default: none)")
    parser.add_argument("--plot", action="store_true",
                        help="Show/save matplotlib plots after training (default: off)")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
