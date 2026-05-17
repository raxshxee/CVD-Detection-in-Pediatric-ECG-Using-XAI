from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
BEATS_ROOT = PROJECT_ROOT / "Preprocessed_Dataset" / "v4_beats"
CHECKPOINT_ROOT = BEATS_ROOT / "checkpoints"
OUTPUT_ROOT = ROOT / "outputs" / "v11_triexpert_fusion"

CLASS_NAMES = [
    "Normal/Non-Cardiac",
    "Structural Heart Disease",
    "Arrhythmia & Electrical",
]
N_CLS = 3
N_LEADS = 12
N_RR = 8
BEAT_LEN = 300
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

V8_DIR = CHECKPOINT_ROOT / "v8_consistent"
V9_DIR = CHECKPOINT_ROOT / "v9_wider"
FEATURES_FILE = BEATS_ROOT / "rf" / "features_v4.npz"
BEATS_FILE = BEATS_ROOT / "beats.dat"
META_FILE = BEATS_ROOT / "beats_meta.npz"


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
    features: np.ndarray
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


@dataclass
class ExpertMetrics:
    name: str
    f1_macro: float
    accuracy: float
    auroc: float


class BeatRRDataset(Dataset):
    def __init__(
        self,
        indices: np.ndarray,
        beats_mmap: np.memmap,
        rr_arr: np.ndarray,
        augment: bool = False,
        tta_strength: float = 1.0,
    ) -> None:
        self.indices = indices
        self.beats = beats_mmap
        self.rr = rr_arr
        self.augment = augment
        self.tta_strength = tta_strength

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        idx = int(self.indices[i])
        beat = self.beats[idx].copy()
        if self.augment:
            beat = augment_beat(beat, strength=self.tta_strength)
        beat_t = torch.tensor(beat.T, dtype=torch.float32)
        rr_t = torch.tensor(self.rr[idx], dtype=torch.float32)
        return beat_t, rr_t


def augment_beat(x: np.ndarray, strength: float = 1.0) -> np.ndarray:
    x = x.copy()
    if np.random.rand() < 0.7:
        x += np.random.normal(0, 0.025 * strength, x.shape).astype(np.float32)
    if np.random.rand() < 0.6:
        x *= np.random.uniform(1.0 - 0.15 * strength, 1.0 + 0.15 * strength)
    if np.random.rand() < 0.35:
        n_drop = np.random.randint(1, 3)
        drops = np.random.choice(N_LEADS, n_drop, replace=False)
        x[:, drops] = 0.0
    if np.random.rand() < 0.4:
        shift = np.random.randint(-15, 15)
        x = np.roll(x, shift, axis=0)
    return x.astype(np.float32)


class ResConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 7, stride: int = 1) -> None:
        super().__init__()
        pad = kernel // 2
        self.c1 = nn.Conv1d(in_ch, out_ch, kernel, stride=stride, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.c2 = nn.Conv1d(out_ch, out_ch, kernel, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.drop = nn.Dropout(0.10)
        if in_ch != out_ch or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.gelu(self.bn1(self.c1(x)))
        out = self.drop(self.bn2(self.c2(out)))
        return F.gelu(out + self.skip(x))


class MorphologyEncoderV8(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(N_LEADS, 64, 9, padding=4, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )
        self.s1 = ResConvBlock(64, 64, 7, 2)
        self.s2 = ResConvBlock(64, 128, 5, 2)
        self.s3 = ResConvBlock(128, 256, 5, 1)
        self.gap = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gap(self.s3(self.s2(self.s1(self.stem(x))))).squeeze(-1)


class RhythmEncoder(nn.Module):
    def __init__(self, rhythm_dim: int = 64) -> None:
        super().__init__()
        self.fc1 = nn.Linear(N_RR, rhythm_dim)
        self.bn1 = nn.BatchNorm1d(rhythm_dim)
        self.fc2 = nn.Linear(rhythm_dim, rhythm_dim)
        self.bn2 = nn.BatchNorm1d(rhythm_dim)
        self.skip = nn.Linear(N_RR, rhythm_dim)

    def forward(self, rr: torch.Tensor) -> torch.Tensor:
        out = F.gelu(self.bn1(self.fc1(rr)))
        out = self.bn2(self.fc2(out))
        return F.gelu(out + self.skip(rr))


class CrossAttentionFusion(nn.Module):
    def __init__(self, morph_dim: int, rhythm_dim: int = 64, heads: int = 2) -> None:
        super().__init__()
        d_head = morph_dim // heads
        self.heads = heads
        self.d_head = d_head
        self.Q = nn.Linear(morph_dim, morph_dim)
        self.K = nn.Linear(rhythm_dim, morph_dim)
        self.V = nn.Linear(rhythm_dim, morph_dim)
        self.proj = nn.Linear(d_head, morph_dim)
        self.norm = nn.LayerNorm(morph_dim)
        self.scale = math.sqrt(d_head)
        self._attn = None

    def forward(self, m: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        batch = m.size(0)
        q = self.Q(m).view(batch, self.heads, self.d_head)
        k = self.K(r).view(batch, self.heads, self.d_head)
        v = self.V(r).view(batch, self.heads, self.d_head)
        w = F.softmax((q * k).sum(-1) / self.scale, dim=1)
        self._attn = w.detach()
        out = (w.unsqueeze(-1) * v).sum(1)
        return self.norm(m + self.proj(out))


class DualBranchV8(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.morph = MorphologyEncoderV8()
        self.rhythm = RhythmEncoder(64)
        self.fusion = CrossAttentionFusion(256, 64, 2)
        self.head = nn.Sequential(
            nn.Linear(256 + 64, 256),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, N_CLS),
        )

    def forward(self, beat: torch.Tensor, rr: torch.Tensor) -> torch.Tensor:
        m = self.morph(beat)
        r = self.rhythm(rr)
        return self.head(torch.cat([self.fusion(m, r), r], dim=1))


class MorphologyEncoderV9(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(N_LEADS, 96, 9, padding=4, bias=False),
            nn.BatchNorm1d(96),
            nn.GELU(),
        )
        self.s1 = ResConvBlock(96, 96, 7, 2)
        self.s2 = ResConvBlock(96, 192, 5, 2)
        self.s3 = ResConvBlock(192, 384, 5, 1)
        self.gap = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gap(self.s3(self.s2(self.s1(self.stem(x))))).squeeze(-1)


class DualBranchV9(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.morph = MorphologyEncoderV9()
        self.rhythm = RhythmEncoder(64)
        self.fusion = CrossAttentionFusion(384, 64, 2)
        self.head = nn.Sequential(
            nn.Linear(384 + 64, 384),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(384, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, N_CLS),
        )

    def forward(self, beat: torch.Tensor, rr: torch.Tensor) -> torch.Tensor:
        m = self.morph(beat)
        r = self.rhythm(rr)
        return self.head(torch.cat([self.fusion(m, r), r], dim=1))


def natural_epoch_key(path: Path) -> int:
    match = re.search(r"_ep(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else -1


def load_assets() -> Assets:
    meta = np.load(META_FILE, allow_pickle=True)
    labels = meta["labels"].astype(np.int32)
    rec_ids = meta["rec_ids"]
    rr = meta["rr_feats"].astype(np.float32)
    rr_norm = StandardScaler().fit_transform(rr).astype(np.float32)
    feat_npz = np.load(FEATURES_FILE, allow_pickle=True)
    features = np.nan_to_num(feat_npz["X"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
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
        features=features,
        unique_recs=unique_recs,
        rec_labels=rec_labels,
        record_to_indices=record_to_indices,
        outer_splits=outer_splits,
    )


def expand_records(record_ids: Iterable[str], record_to_indices: dict[str, np.ndarray]) -> np.ndarray:
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


def make_loader(indices: np.ndarray, assets: Assets, batch_size: int, augment: bool = False, tta_strength: float = 0.7) -> DataLoader:
    ds = BeatRRDataset(indices, assets.beats, assets.rr_norm, augment=augment, tta_strength=tta_strength)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)


@torch.no_grad()
def predict_probabilities(models: list[nn.Module], loader: DataLoader) -> np.ndarray:
    prob_sum = None
    for model in models:
        model.eval()
        batch_probs: list[np.ndarray] = []
        for beat, rr in loader:
            beat = beat.to(DEVICE)
            rr = rr.to(DEVICE)
            probs = F.softmax(model(beat, rr), dim=1).cpu().numpy()
            batch_probs.append(probs)
        model_probs = np.concatenate(batch_probs, axis=0)
        if prob_sum is None:
            prob_sum = model_probs.astype(np.float64)
        else:
            prob_sum += model_probs
    if prob_sum is None:
        raise RuntimeError("No model probabilities were produced.")
    return (prob_sum / len(models)).astype(np.float32)


@torch.no_grad()
def predict_probabilities_v9_tta(
    models: list[nn.Module],
    indices: np.ndarray,
    assets: Assets,
    batch_size: int,
    n_tta: int,
) -> np.ndarray:
    pass_probs = []
    base_loader = make_loader(indices, assets, batch_size=batch_size, augment=False)
    pass_probs.append(predict_probabilities(models, base_loader))
    for _ in range(max(0, n_tta - 1)):
        aug_loader = make_loader(indices, assets, batch_size=batch_size, augment=True, tta_strength=0.7)
        pass_probs.append(predict_probabilities(models, aug_loader))
    return np.mean(pass_probs, axis=0).astype(np.float32)


def load_v8_models(fold_name: str) -> list[nn.Module]:
    ckpts = sorted(V8_DIR.glob(f"v8_{fold_name}_ep*.pt"), key=natural_epoch_key)
    models = []
    for ckpt in ckpts:
        model = DualBranchV8().to(DEVICE)
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        model.eval()
        models.append(model)
    if not models:
        raise FileNotFoundError(f"No V8 checkpoints found for {fold_name} in {V8_DIR}")
    return models


def load_v9_models(fold_name: str) -> list[nn.Module]:
    ckpts = sorted(V9_DIR.glob(f"v9_{fold_name}_ep*.pt"), key=natural_epoch_key)
    models = []
    for ckpt in ckpts:
        model = DualBranchV9().to(DEVICE)
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        model.eval()
        models.append(model)
    if not models:
        raise FileNotFoundError(f"No V9 checkpoints found for {fold_name} in {V9_DIR}")
    return models


def train_xgb_expert(
    fold: FoldIndices,
    assets: Assets,
    out_dir: Path,
    seed: int,
) -> XGBClassifier:
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=N_CLS,
        eval_metric="mlogloss",
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        min_child_weight=2,
        subsample=0.85,
        colsample_bytree=0.80,
        reg_lambda=2.0,
        tree_method="hist",
        random_state=seed,
        n_jobs=max(1, (os.cpu_count() or 4) - 1),
    )
    X_train = assets.features[fold.train_idx]
    y_train = assets.labels[fold.train_idx]
    X_calib = assets.features[fold.calib_idx]
    y_calib = assets.labels[fold.calib_idx]
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_calib, y_calib)],
        verbose=False,
    )
    joblib.dump(model, out_dir / f"{fold.fold_name}_xgb.pkl")
    return model


def evaluate_predictions(y_true: np.ndarray, probs: np.ndarray, name: str) -> ExpertMetrics:
    preds = probs.argmax(axis=1)
    return ExpertMetrics(
        name=name,
        f1_macro=float(f1_score(y_true, preds, average="macro", zero_division=0)),
        accuracy=float(accuracy_score(y_true, preds)),
        auroc=float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro")),
    )


def prob_entropy(probs: np.ndarray) -> np.ndarray:
    eps = 1e-8
    return -(probs * np.log(probs + eps)).sum(axis=1, keepdims=True)


def prob_margin(probs: np.ndarray) -> np.ndarray:
    top2 = np.partition(probs, kth=-2, axis=1)[:, -2:]
    return (top2[:, 1] - top2[:, 0]).reshape(-1, 1)


def build_meta_features(v8_probs: np.ndarray, v9_probs: np.ndarray, xgb_probs: np.ndarray) -> np.ndarray:
    parts = [
        v8_probs,
        v9_probs,
        xgb_probs,
        prob_entropy(v8_probs),
        prob_entropy(v9_probs),
        prob_entropy(xgb_probs),
        prob_margin(v8_probs),
        prob_margin(v9_probs),
        prob_margin(xgb_probs),
    ]
    return np.concatenate(parts, axis=1).astype(np.float32)


def grid_search_fusion(
    y_true: np.ndarray,
    candidates: dict[str, np.ndarray],
) -> tuple[str, dict[str, float], np.ndarray, float]:
    names = list(candidates.keys())
    best_name = ""
    best_weights: dict[str, float] = {}
    best_probs = None
    best_f1 = -1.0

    simplex = np.linspace(0.0, 1.0, 21)
    for w0 in simplex:
        for w1 in simplex:
            w2 = 1.0 - w0 - w1
            if w2 < 0 or w2 > 1:
                continue
            weights = np.array([w0, w1, w2], dtype=np.float32)
            if np.count_nonzero(weights) == 0:
                continue
            probs = sum(weights[i] * candidates[names[i]] for i in range(3))
            f1 = f1_score(y_true, probs.argmax(axis=1), average="macro", zero_division=0)
            if f1 > best_f1:
                best_f1 = float(f1)
                best_name = "weighted_grid"
                best_weights = {names[i]: float(weights[i]) for i in range(3)}
                best_probs = probs.astype(np.float32)

    if best_probs is None:
        raise RuntimeError("Fusion grid search did not produce any candidate.")
    return best_name, best_weights, best_probs, best_f1


def fit_meta_learner(
    meta_train: np.ndarray,
    y_train: np.ndarray,
    meta_test: np.ndarray,
) -> tuple[LogisticRegression, StandardScaler, np.ndarray]:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(meta_train)
    X_test = scaler.transform(meta_test)
    model = LogisticRegression(
        max_iter=1500,
        class_weight="balanced",
        C=0.5,
        solver="lbfgs",
        random_state=42,
    )
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test).astype(np.float32)
    return model, scaler, probs


def choose_fusion(
    calib_labels: np.ndarray,
    calib_probs: dict[str, np.ndarray],
    test_probs: dict[str, np.ndarray],
) -> tuple[str, dict[str, float], np.ndarray, float]:
    meta_calib = build_meta_features(calib_probs["v8"], calib_probs["v9"], calib_probs["xgb"])
    meta_test = build_meta_features(test_probs["v8"], test_probs["v9"], test_probs["xgb"])
    meta_model, meta_scaler, meta_probs_test = fit_meta_learner(meta_calib, calib_labels, meta_test)
    meta_probs_calib = meta_model.predict_proba(meta_scaler.transform(meta_calib)).astype(np.float32)

    candidate_scores: list[tuple[str, float]] = []
    for name, probs in calib_probs.items():
        f1 = f1_score(calib_labels, probs.argmax(axis=1), average="macro", zero_division=0)
        candidate_scores.append((name, float(f1)))

    weighted_name, weighted_weights, weighted_calib_probs, weighted_f1 = grid_search_fusion(
        calib_labels,
        {"v8": calib_probs["v8"], "v9": calib_probs["v9"], "xgb": calib_probs["xgb"]},
    )
    weighted_test_probs = (
        weighted_weights["v8"] * test_probs["v8"]
        + weighted_weights["v9"] * test_probs["v9"]
        + weighted_weights["xgb"] * test_probs["xgb"]
    ).astype(np.float32)
    candidate_scores.append((weighted_name, weighted_f1))

    meta_f1 = f1_score(calib_labels, meta_probs_calib.argmax(axis=1), average="macro", zero_division=0)
    candidate_scores.append(("meta_logreg", float(meta_f1)))

    best_name, _ = max(candidate_scores, key=lambda item: item[1])
    if best_name in calib_probs:
        return best_name, {best_name: 1.0}, test_probs[best_name], dict(candidate_scores)[best_name]
    if best_name == "weighted_grid":
        return best_name, weighted_weights, weighted_test_probs, weighted_f1
    return "meta_logreg", {"uses_meta_features": 1.0}, meta_probs_test, float(meta_f1)


def run_fold(
    assets: Assets,
    fold_number: int,
    batch_size: int,
    v9_tta: int,
    seed: int,
) -> dict:
    fold = make_fold_indices(assets, fold_number)
    out_dir = OUTPUT_ROOT / fold.fold_name
    out_dir.mkdir(parents=True, exist_ok=True)

    log(f"\n{'=' * 78}")
    log(f"{fold.fold_name.upper()} | train beats={len(fold.train_idx):,} | calib beats={len(fold.calib_idx):,} | test beats={len(fold.test_idx):,}")
    log(f"{fold.fold_name.upper()} | train recs={len(fold.train_recs):,} | calib recs={len(fold.calib_recs):,} | test recs={len(fold.test_recs):,}")

    xgb = train_xgb_expert(fold, assets, out_dir, seed)
    xgb_calib = xgb.predict_proba(assets.features[fold.calib_idx]).astype(np.float32)
    xgb_test = xgb.predict_proba(assets.features[fold.test_idx]).astype(np.float32)

    log(f"{fold.fold_name}: loading V8 checkpoints...")
    v8_models = load_v8_models(fold.fold_name)
    log(f"{fold.fold_name}: loading V9 checkpoints...")
    v9_models = load_v9_models(fold.fold_name)

    calib_loader = make_loader(fold.calib_idx, assets, batch_size=batch_size, augment=False)
    test_loader = make_loader(fold.test_idx, assets, batch_size=batch_size, augment=False)

    start = time.time()
    v8_calib = predict_probabilities(v8_models, calib_loader)
    v8_test = predict_probabilities(v8_models, test_loader)
    log(f"{fold.fold_name}: V8 inference complete in {time.time() - start:.1f}s")

    start = time.time()
    v9_calib = predict_probabilities_v9_tta(v9_models, fold.calib_idx, assets, batch_size=batch_size, n_tta=v9_tta)
    v9_test = predict_probabilities_v9_tta(v9_models, fold.test_idx, assets, batch_size=batch_size, n_tta=v9_tta)
    log(f"{fold.fold_name}: V9 TTA inference complete in {time.time() - start:.1f}s")

    calib_labels = assets.labels[fold.calib_idx]
    test_labels = assets.labels[fold.test_idx]

    calib_probs = {"v8": v8_calib, "v9": v9_calib, "xgb": xgb_calib}
    test_probs = {"v8": v8_test, "v9": v9_test, "xgb": xgb_test}
    chosen_name, chosen_params, fused_test_probs, calib_best_f1 = choose_fusion(calib_labels, calib_probs, test_probs)

    fold_summary = {
        "fold": fold.fold_name,
        "calibration_choice": chosen_name,
        "calibration_f1": calib_best_f1,
        "choice_params": chosen_params,
        "experts": {
            "v8": asdict(evaluate_predictions(test_labels, v8_test, "v8")),
            "v9": asdict(evaluate_predictions(test_labels, v9_test, "v9")),
            "xgb": asdict(evaluate_predictions(test_labels, xgb_test, "xgb")),
            "fusion": asdict(evaluate_predictions(test_labels, fused_test_probs, chosen_name)),
        },
    }

    (out_dir / "fold_summary.json").write_text(json.dumps(fold_summary, indent=2), encoding="utf-8")
    np.save(out_dir / "fusion_test_probs.npy", fused_test_probs)
    np.save(out_dir / "test_labels.npy", test_labels)
    log(
        f"{fold.fold_name}: fusion={chosen_name} | "
        f"F1={fold_summary['experts']['fusion']['f1_macro']:.4f} | "
        f"Acc={fold_summary['experts']['fusion']['accuracy']:.4f} | "
        f"AUROC={fold_summary['experts']['fusion']['auroc']:.4f}"
    )
    return fold_summary


def summarise_runs(fold_summaries: list[dict]) -> dict:
    fusion_metrics = [item["experts"]["fusion"] for item in fold_summaries]
    v9_metrics = [item["experts"]["v9"] for item in fold_summaries]
    v8_metrics = [item["experts"]["v8"] for item in fold_summaries]
    xgb_metrics = [item["experts"]["xgb"] for item in fold_summaries]

    def block(name: str, rows: list[dict]) -> dict:
        return {
            "name": name,
            "f1_mean": float(np.mean([row["f1_macro"] for row in rows])),
            "f1_std": float(np.std([row["f1_macro"] for row in rows])),
            "acc_mean": float(np.mean([row["accuracy"] for row in rows])),
            "auroc_mean": float(np.mean([row["auroc"] for row in rows])),
        }

    return {
        "num_folds": len(fold_summaries),
        "v11_fusion": block("v11_fusion", fusion_metrics),
        "v9_expert": block("v9_expert", v9_metrics),
        "v8_expert": block("v8_expert", v8_metrics),
        "xgb_expert": block("xgb_expert", xgb_metrics),
        "folds": fold_summaries,
    }


def parse_folds(text: str) -> list[int]:
    folds = []
    for part in text.split(","):
        value = int(part.strip())
        if value < 1 or value > 5:
            raise ValueError("Fold numbers must be between 1 and 5.")
        folds.append(value)
    return sorted(set(folds))


def main() -> None:
    parser = argparse.ArgumentParser(description="V11 patient-aware tri-expert fusion for pediatric ECG.")
    parser.add_argument("--folds", type=str, default="1,2,3,4,5", help="Comma-separated fold numbers to run.")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--v9-tta", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    log(f"Device: {DEVICE}")
    log("Loading assets...")
    assets = load_assets()
    folds = parse_folds(args.folds)
    log(f"Running folds: {folds}")

    fold_summaries = []
    for fold_number in folds:
        fold_summaries.append(
            run_fold(
                assets=assets,
                fold_number=fold_number,
                batch_size=args.batch_size,
                v9_tta=args.v9_tta,
                seed=args.seed + fold_number,
            )
        )

    summary = summarise_runs(fold_summaries)
    summary_path = OUTPUT_ROOT / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    summary_rows = []
    for key in ("v11_fusion", "v9_expert", "v8_expert", "xgb_expert"):
        block = summary[key]
        summary_rows.append(
            {
                "model": block["name"],
                "f1_mean": block["f1_mean"],
                "f1_std": block["f1_std"],
                "acc_mean": block["acc_mean"],
                "auroc_mean": block["auroc_mean"],
            }
        )
    pd.DataFrame(summary_rows).to_csv(OUTPUT_ROOT / "summary.csv", index=False)

    fusion = summary["v11_fusion"]
    v9 = summary["v9_expert"]
    log("\n" + "=" * 78)
    log(f"V11 Fusion | Macro F1={fusion['f1_mean']:.4f} +/- {fusion['f1_std']:.4f} | Acc={fusion['acc_mean']:.4f} | AUROC={fusion['auroc_mean']:.4f}")
    log(f"V9 Expert  | Macro F1={v9['f1_mean']:.4f} +/- {v9['f1_std']:.4f} | Acc={v9['acc_mean']:.4f} | AUROC={v9['auroc_mean']:.4f}")
    log(f"Delta F1   | {fusion['f1_mean'] - v9['f1_mean']:+.4f}")
    log(f"Artifacts  | {OUTPUT_ROOT}")
    log("=" * 78)


if __name__ == "__main__":
    main()
