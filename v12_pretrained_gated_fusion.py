from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from ecglib.models.model_builder import create_model
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
BEATS_ROOT = PROJECT_ROOT / "Preprocessed_Dataset" / "v4_beats"
OUTPUT_ROOT = ROOT / "outputs" / "v12_pretrained_gated_fusion"
CACHE_DIR = OUTPUT_ROOT / "cache"

BEATS_FILE = BEATS_ROOT / "beats.dat"
META_FILE = BEATS_ROOT / "beats_meta.npz"

CLASS_NAMES = [
    "Normal/Non-Cardiac",
    "Structural Heart Disease",
    "Arrhythmia & Electrical",
]
N_CLS = 3
N_LEADS = 12
N_RR = 8
BEAT_LEN = 300
EMBED_DIM = 2048
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def log(message: str) -> None:
    print(message, flush=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class Assets:
    beats: np.memmap
    labels: np.ndarray
    rec_ids: np.ndarray
    rr_norm: np.ndarray
    unique_recs: np.ndarray
    rec_labels: np.ndarray
    record_to_indices: dict[str, np.ndarray]
    outer_splits: list[tuple[np.ndarray, np.ndarray]]


@dataclass
class FoldIndices:
    fold_name: str
    train_idx: np.ndarray
    calib_idx: np.ndarray
    test_idx: np.ndarray
    train_recs: np.ndarray
    calib_recs: np.ndarray
    test_recs: np.ndarray


class BeatOnlyDataset(Dataset):
    def __init__(self, beats_mmap: np.memmap) -> None:
        self.beats = beats_mmap

    def __len__(self) -> int:
        return len(self.beats)

    def __getitem__(self, i: int) -> torch.Tensor:
        beat = self.beats[i].copy().T
        return torch.tensor(beat, dtype=torch.float32)


class EmbeddingRRDataset(Dataset):
    def __init__(self, indices: np.ndarray, embeddings: np.memmap, rr: np.ndarray, labels: np.ndarray) -> None:
        self.indices = indices
        self.embeddings = embeddings
        self.rr = rr
        self.labels = labels

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = int(self.indices[i])
        emb = torch.tensor(self.embeddings[idx].astype(np.float32), dtype=torch.float32)
        rr = torch.tensor(self.rr[idx], dtype=torch.float32)
        y = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return emb, rr, y


def load_assets() -> Assets:
    meta = np.load(META_FILE, allow_pickle=True)
    labels = meta["labels"].astype(np.int32)
    rec_ids = meta["rec_ids"]
    rr = meta["rr_feats"].astype(np.float32)
    rr_norm = StandardScaler().fit_transform(rr).astype(np.float32)
    beats = np.memmap(BEATS_FILE, dtype="float32", mode="r", shape=(len(labels), BEAT_LEN, N_LEADS))

    unique_recs = np.unique(rec_ids)
    rec_labels = np.array(
        [int(np.argmax(np.bincount(labels[rec_ids == record], minlength=N_CLS))) for record in unique_recs],
        dtype=np.int32,
    )
    record_to_indices = {record: np.where(rec_ids == record)[0] for record in unique_recs}
    outer_splitter = StratifiedShuffleSplit(n_splits=5, test_size=0.20, random_state=42)
    outer_splits = list(outer_splitter.split(unique_recs, rec_labels))

    return Assets(
        beats=beats,
        labels=labels,
        rec_ids=rec_ids,
        rr_norm=rr_norm,
        unique_recs=unique_recs,
        rec_labels=rec_labels,
        record_to_indices=record_to_indices,
        outer_splits=outer_splits,
    )


def expand_records(record_ids: np.ndarray, record_to_indices: dict[str, np.ndarray]) -> np.ndarray:
    arrays = [record_to_indices[str(record)] for record in record_ids]
    return np.concatenate(arrays).astype(np.int64)


def make_fold_indices(assets: Assets, fold_number: int) -> FoldIndices:
    outer_train_idx, outer_test_idx = assets.outer_splits[fold_number - 1]
    outer_train_recs = assets.unique_recs[outer_train_idx]
    outer_test_recs = assets.unique_recs[outer_test_idx]
    outer_train_labels = assets.rec_labels[outer_train_idx]

    inner_splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=fold_number - 1)
    inner_train_rel, inner_calib_rel = next(inner_splitter.split(outer_train_recs, outer_train_labels))
    train_recs = outer_train_recs[inner_train_rel]
    calib_recs = outer_train_recs[inner_calib_rel]

    return FoldIndices(
        fold_name=f"fold_{fold_number}",
        train_idx=expand_records(train_recs, assets.record_to_indices),
        calib_idx=expand_records(calib_recs, assets.record_to_indices),
        test_idx=expand_records(outer_test_recs, assets.record_to_indices),
        train_recs=train_recs,
        calib_recs=calib_recs,
        test_recs=outer_test_recs,
    )


class PretrainedResNetEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        base = create_model(
            model_name="resnet1d18",
            pathology="AFIB",
            pretrained=True,
            leads_count=12,
            num_classes=1,
        )
        self.cnn, _ = base.get_cnn()
        self.pool = nn.AdaptiveAvgPool1d(1)
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.cnn(x)
        return self.pool(z).squeeze(-1)


class GatedEmbeddingHead(nn.Module):
    def __init__(self, embed_dim: int = EMBED_DIM, rr_dim: int = 64) -> None:
        super().__init__()
        self.rr = nn.Sequential(
            nn.Linear(N_RR, rr_dim),
            nn.BatchNorm1d(rr_dim),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(rr_dim, rr_dim),
            nn.GELU(),
        )
        self.gate = nn.Sequential(
            nn.Linear(rr_dim, embed_dim),
            nn.Sigmoid(),
        )
        self.head = nn.Sequential(
            nn.Linear(embed_dim + rr_dim, 256),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(128, N_CLS),
        )

    def forward(self, emb: torch.Tensor, rr: torch.Tensor) -> torch.Tensor:
        rr_z = self.rr(rr)
        gated = emb * self.gate(rr_z)
        return self.head(torch.cat([gated, rr_z], dim=1))


def build_embedding_cache(assets: Assets, batch_size: int) -> np.memmap:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / "resnet18_afib_embeddings.f16.dat"
    if cache_file.exists():
        log(f"Using cached embeddings: {cache_file}")
        return np.memmap(cache_file, dtype="float16", mode="r", shape=(len(assets.labels), EMBED_DIM))

    log("Building pretrained embedding cache...")
    encoder = PretrainedResNetEncoder().to(DEVICE)
    encoder.eval()
    loader = DataLoader(BeatOnlyDataset(assets.beats), batch_size=batch_size, shuffle=False, num_workers=0)
    mem = np.memmap(cache_file, dtype="float16", mode="w+", shape=(len(assets.labels), EMBED_DIM))
    start = 0
    with torch.no_grad():
        for batch_i, beats in enumerate(loader, start=1):
            beats = beats.to(DEVICE)
            emb = encoder(beats).cpu().numpy().astype(np.float16)
            stop = start + len(emb)
            mem[start:stop] = emb
            start = stop
            if batch_i % 20 == 0 or batch_i == len(loader):
                log(f"  cached {stop:,}/{len(assets.labels):,} beats")
    mem.flush()
    return np.memmap(cache_file, dtype="float16", mode="r", shape=(len(assets.labels), EMBED_DIM))


def make_loader(indices: np.ndarray, embeddings: np.memmap, assets: Assets, batch_size: int, shuffle: bool) -> DataLoader:
    ds = EmbeddingRRDataset(indices, embeddings, assets.rr_norm, assets.labels)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def train_model(
    model: nn.Module,
    tr_loader: DataLoader,
    vl_loader: DataLoader,
    max_epochs: int,
    lr: float,
    patience: int,
) -> tuple[nn.Module, list[dict]]:
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=2e-4)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    best_state = None
    best_loss = float("inf")
    bad = 0
    history = []

    for epoch in range(max_epochs):
        model.train()
        tl, cor, tot = 0.0, 0, 0
        for emb, rr, yb in tr_loader:
            emb = emb.to(DEVICE)
            rr = rr.to(DEVICE)
            yb = yb.to(DEVICE)
            opt.zero_grad()
            logits = model(emb, rr)
            loss = crit(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += float(loss.item())
            cor += int((logits.argmax(1) == yb).sum().item())
            tot += int(yb.size(0))

        model.eval()
        vl = 0.0
        all_p, all_y = [], []
        with torch.no_grad():
            for emb, rr, yb in vl_loader:
                emb = emb.to(DEVICE)
                rr = rr.to(DEVICE)
                yb = yb.to(DEVICE)
                logits = model(emb, rr)
                loss = crit(logits, yb)
                probs = F.softmax(logits, dim=1)
                vl += float(loss.item())
                all_p.append(probs.cpu().numpy())
                all_y.append(yb.cpu().numpy())
        probs = np.concatenate(all_p, axis=0)
        labels = np.concatenate(all_y, axis=0)
        val_f1 = f1_score(labels, probs.argmax(1), average="macro", zero_division=0)
        val_acc = accuracy_score(labels, probs.argmax(1))
        val_auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
        mean_vl = vl / max(1, len(vl_loader))
        history.append(
            {
                "epoch": epoch + 1,
                "tr_loss": tl / max(1, len(tr_loader)),
                "tr_acc": cor / max(1, tot),
                "val_loss": mean_vl,
                "val_acc": val_acc,
                "val_f1": val_f1,
                "val_auroc": val_auc,
            }
        )
        if mean_vl < best_loss - 1e-4:
            best_loss = mean_vl
            bad = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1

        log(
            f"  ep {epoch + 1:02d}/{max_epochs} | tr_loss={history[-1]['tr_loss']:.4f} "
            f"| val_loss={mean_vl:.4f} | val_f1={val_f1:.4f}"
        )
        if bad >= patience:
            log(f"  early stop at epoch {epoch + 1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    model.eval()
    all_p, all_y = [], []
    for emb, rr, yb in loader:
        emb = emb.to(DEVICE)
        rr = rr.to(DEVICE)
        logits = model(emb, rr)
        probs = F.softmax(logits, dim=1)
        all_p.append(probs.cpu().numpy())
        all_y.append(yb.numpy())
    probs = np.concatenate(all_p, axis=0)
    labels = np.concatenate(all_y, axis=0)
    return (
        f1_score(labels, probs.argmax(1), average="macro", zero_division=0),
        accuracy_score(labels, probs.argmax(1)),
        roc_auc_score(labels, probs, multi_class="ovr", average="macro"),
        probs,
        labels,
    )


def run_fold(
    assets: Assets,
    embeddings: np.memmap,
    fold_number: int,
    batch_size: int,
    lr: float,
    max_epochs: int,
    patience: int,
) -> dict:
    fold = make_fold_indices(assets, fold_number)
    out_dir = OUTPUT_ROOT / fold.fold_name
    out_dir.mkdir(parents=True, exist_ok=True)

    log(
        f"\n{fold.fold_name.upper()} | train beats={len(fold.train_idx):,} | "
        f"calib beats={len(fold.calib_idx):,} | test beats={len(fold.test_idx):,}"
    )

    tr_loader = make_loader(fold.train_idx, embeddings, assets, batch_size, True)
    vl_loader = make_loader(fold.calib_idx, embeddings, assets, batch_size * 2, False)
    te_loader = make_loader(fold.test_idx, embeddings, assets, batch_size * 2, False)

    model = GatedEmbeddingHead().to(DEVICE)
    model, history = train_model(model, tr_loader, vl_loader, max_epochs=max_epochs, lr=lr, patience=patience)

    f1, acc, auroc, probs, labels = evaluate(model, te_loader)
    summary = {
        "fold": fold.fold_name,
        "f1_macro": float(f1),
        "accuracy": float(acc),
        "auroc": float(auroc),
        "history": history,
        "train_recs": int(len(fold.train_recs)),
        "calib_recs": int(len(fold.calib_recs)),
        "test_recs": int(len(fold.test_recs)),
    }
    (out_dir / "fold_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.save(out_dir / "test_probs.npy", probs)
    np.save(out_dir / "test_labels.npy", labels)
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    log(f"  test: F1={f1:.4f} | Acc={acc:.4f} | AUROC={auroc:.4f}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrained ECG embedding plus RR-gated pediatric classifier.")
    parser.add_argument("--folds", type=str, default="1,2,3,4,5")
    parser.add_argument("--embed-batch-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-epochs", type=int, default=18)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    assets = load_assets()
    embeddings = build_embedding_cache(assets, batch_size=args.embed_batch_size)
    folds = [int(x.strip()) for x in args.folds.split(",") if x.strip()]

    summaries = []
    for fold_number in folds:
        summaries.append(
            run_fold(
                assets=assets,
                embeddings=embeddings,
                fold_number=fold_number,
                batch_size=args.batch_size,
                lr=args.lr,
                max_epochs=args.max_epochs,
                patience=args.patience,
            )
        )

    df = pd.DataFrame(
        [
            {"fold": item["fold"], "f1_macro": item["f1_macro"], "accuracy": item["accuracy"], "auroc": item["auroc"]}
            for item in summaries
        ]
    )
    df.to_csv(OUTPUT_ROOT / "summary.csv", index=False)
    final = {
        "num_folds": len(summaries),
        "f1_mean": float(df["f1_macro"].mean()),
        "f1_std": float(df["f1_macro"].std(ddof=0)),
        "acc_mean": float(df["accuracy"].mean()),
        "auroc_mean": float(df["auroc"].mean()),
        "folds": summaries,
    }
    (OUTPUT_ROOT / "summary.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    log("\n" + json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
