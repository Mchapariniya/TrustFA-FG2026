#!/usr/bin/env python3
"""
==========================================================================
Ablation Study: Dynamic Features vs Appearance Features
==========================================================================

This ablation directly addresses the reviewer question:
  "How do you know the model is not using appearance-based cues
   from the raw pixel input rather than motion dynamics?"

We train and evaluate FOUR conditions that isolate what information
each feature type carries:

  Condition A — DIFF (proposed):
    feat_{t+1} - feat_t   ← inter-frame difference in deep feature space
    Encodes ONLY motion. A static face gives zero signal.

  Condition B — RAW_FEAT:
    feat_t                ← raw deep features, no subtraction
    Encodes appearance + motion. If A ≈ B, appearance doesn't help.
    If A > B, the appearance signal is HURTING (it's a distractor).

  Condition C — PIXEL_DIFF:
    frame_{t+1} - frame_t  ← raw pixel-level optical flow proxy
    Bypasses the learned ConvStack. Tests whether deep features
    are needed on top of raw motion signal.

  Condition D — STATIC (appearance oracle):
    Single frame (T=1), no temporal information at all.
    Should perform near chance (AUC ≈ 0.5) if appearance is truly
    uninformative. If D is significantly above 0.5, the network
    CAN exploit appearance — which would weaken our claim.

EXPECTED OUTCOME (supporting the dynamic features claim):
  A (DIFF) >> B (RAW_FEAT) ≈ D (STATIC) > C (PIXEL_DIFF)

  Specifically:
  - A >> D proves temporal dynamics matter, not static appearance
  - A ≥ B proves that adding appearance signal does NOT help
    (or actively hurts, if B < A)
  - A > C proves that learning a deep feature space before differencing
    (MOL's F5C backbone) improves over naive pixel differences

TABLE FORMAT (for paper Table):
  ┌─────────────────────────────┬──────────┬──────────────────────────────┐
  │ Condition                   │ AUC      │ What it shows                │
  ├─────────────────────────────┼──────────┼──────────────────────────────┤
  │ A: Deep diff (proposed)     │ 0.87x    │ Full model                   │
  │ B: Raw deep features        │ ~0.8x    │ Appearance hurts/neutral     │
  │ C: Pixel-level diff         │ ~0.7x    │ Deep features needed         │
  │ D: Static frame (no motion) │ ~0.5x    │ Appearance alone ≈ chance    │
  └─────────────────────────────┴──────────┴──────────────────────────────┘

USAGE
-----
  # Run all 4 conditions (full ablation):
  CUDA_VISIBLE_DEVICES=0 python ablation_dynamic_vs_appearance.py \\
      --nvfair_split_root NVFAIR_split \\
      --frames_root       nvfair_frames \\
      --output_dir        outputs/ablation_dynamic \\
      --results_dir       results/ablation_dynamic \\
      --num_epochs 30 \\
      --steps_per_epoch 200

  # Run only specific conditions:
  python ablation_dynamic_vs_appearance.py \\
      --nvfair_split_root NVFAIR_split \\
      --frames_root       nvfair_frames \\
      --conditions DIFF RAW_FEAT STATIC

  # Plot only (if checkpoints already exist):
  python ablation_dynamic_vs_appearance.py \\
      --results_dir results/ablation_dynamic \\
      --plot_only
"""

import argparse
import json
import logging
import math
import os
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Reproduces the F5C representation ablation (Table VI: FEAT_DIFF / PIXEL_DIFF /
# RAW_FEAT / STATIC). This script is NOT self-contained: it reuses the original
# F5C backbone modules and expects these symbols to be importable from
# `train_mol_avatar`:
#     ConvStack, FCC, CCC, get_graph_feature, TemporalIdentityHead,
#     NVFAIRFrameDataset, IdentityBatchSampler, SupConLoss,
#     compute_auc_paper_method, extract_all_embeddings, collate_fn, set_seed
#
# RECOMMENDED (license-safe, runs immediately): reuse the self-contained F5C
# definitions already shipped in src/feat_diff.py, e.g. add the repo root to
# PYTHONPATH and replace the import below with:
#       from src.feat_diff import (
#           ConvStack, FCC, CCC, get_graph_feature, TemporalIdentityHead,
#           NVFAIRFrameDataset, IdentityBatchSampler, SupConLoss,
#           compute_auc_paper_method, extract_all_embeddings, collate_fn, set_seed)
#
# ALTERNATIVE: provide mol_model.py + a matching train_mol_avatar.py that export
# these symbols. The backbone originates from the MOL micro-expression model
# (Shao et al., TPAMI 2025; https://github.com/CYF-cuber/MOL — Conv_stack / FCC /
# CCC modules). CAUTION: that repository ships NO license, so redistributing
# MOL-derived code in a public release is not permitted without the authors'
# consent. The self-contained re-implementation in src/feat_diff.py avoids this.
# -----------------------------------------------------------------------------
import os as _os
import sys as _sys
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)

try:
    from src.feat_diff import (
        ConvStack, FCC, CCC, get_graph_feature,
        TemporalIdentityHead,
        NVFAIRFrameDataset, IdentityBatchSampler,
        SupConLoss, compute_auc_paper_method,
        extract_all_embeddings, collate_fn, set_seed,
    )
except ImportError as e:
    raise ImportError(
        f"Cannot import the F5C backbone from src.feat_diff: {e}\n"
        "Run from the repository root (or add it to PYTHONPATH)."
    )

try:
    from sklearn.metrics import roc_auc_score, roc_curve
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as e:
    raise ImportError(f"Missing dependency: {e}")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# All four ablation conditions
ALL_CONDITIONS = ["DIFF", "RAW_FEAT", "PIXEL_DIFF", "STATIC"]

CONDITION_DESCRIPTIONS = {
    "DIFF":       "Deep diff (proposed) — feat_{t+1} − feat_t",
    "RAW_FEAT":   "Raw deep features  — feat_t (no subtraction)",
    "PIXEL_DIFF": "Pixel-level diff   — frame_{t+1} − frame_t",
    "STATIC":     "Static frame       — single frame, no motion",
}

CONDITION_COLORS = {
    "DIFF":       "#2E86AB",   # blue   — our method
    "RAW_FEAT":   "#E84855",   # red    — appearance contaminated
    "PIXEL_DIFF": "#F18F01",   # orange — shallow motion
    "STATIC":     "#7D8CA3",   # grey   — no motion at all
}


###############################################################################
#  PART 1 — FOUR MODEL VARIANTS
###############################################################################

class MOLBackbone_DIFF(nn.Module):
    """
    PROPOSED MODEL (Condition A).
    Inter-frame differences in deep feature space.
    feat_{t+1} - feat_t  → (B, 128, 16, 16, T-1)
    """
    def __init__(self, neighbor_k=4):
        super().__init__()
        self.conv_stack = ConvStack()
        self.fcc = FCC(dim=64, meta_kernel_size=32)
        self.ccc = CCC()
        self.k = neighbor_k

    def _frame_feat(self, frame):
        f = self.conv_stack(frame)
        f = self.fcc(f)
        B, C, H, W = f.shape
        fl = f.reshape(B, C, H * W)
        gf = get_graph_feature(fl, k=self.k)
        ef = self.ccc(fl, gf)
        return (fl + ef).reshape(B, C, H, W)

    def forward(self, frames):
        """frames: (B, T, 1, 128, 128)  →  (B, 128, 16, 16, T-1)"""
        B, T, C, H, W = frames.shape
        diffs, prev = [], None
        for t in range(T):
            cur = self._frame_feat(frames[:, t])
            if prev is not None:
                diffs.append(cur - prev)   # ← THE KEY OPERATION
            prev = cur
        return torch.stack(diffs, dim=-1)  # (B, 128, 16, 16, T-1)


class MOLBackbone_RAW_FEAT(nn.Module):
    """
    ABLATION B: Raw deep features — no inter-frame subtraction.
    Passes feat_t directly. Encodes appearance + motion.
    Uses T frames stacked (same temporal depth as DIFF uses T-1).
    """
    def __init__(self, neighbor_k=4):
        super().__init__()
        self.conv_stack = ConvStack()
        self.fcc = FCC(dim=64, meta_kernel_size=32)
        self.ccc = CCC()
        self.k = neighbor_k

    def _frame_feat(self, frame):
        f = self.conv_stack(frame)
        f = self.fcc(f)
        B, C, H, W = f.shape
        fl = f.reshape(B, C, H * W)
        gf = get_graph_feature(fl, k=self.k)
        ef = self.ccc(fl, gf)
        return (fl + ef).reshape(B, C, H, W)

    def forward(self, frames):
        """frames: (B, T, 1, 128, 128)  →  (B, 128, 16, 16, T)"""
        B, T, C, H, W = frames.shape
        feats = []
        for t in range(T):
            feats.append(self._frame_feat(frames[:, t]))   # NO subtraction
        return torch.stack(feats, dim=-1)   # (B, 128, 16, 16, T)


class MOLBackbone_PIXEL_DIFF(nn.Module):
    """
    ABLATION C: Pixel-level inter-frame differences.
    Computes frame_{t+1} - frame_t in pixel space, THEN encodes.
    Tests whether the learned deep feature space adds value over
    naive optical-flow-like differences.
    """
    def __init__(self, neighbor_k=4):
        super().__init__()
        # ConvStack for the pixel-diff input (same architecture)
        self.conv_stack = ConvStack()
        self.fcc = FCC(dim=64, meta_kernel_size=32)
        self.ccc = CCC()
        self.k = neighbor_k

    def _encode(self, pixel_diff):
        """pixel_diff: (B, 1, 128, 128) → (B, 128, 16, 16)"""
        f = self.conv_stack(pixel_diff)
        f = self.fcc(f)
        B, C, H, W = f.shape
        fl = f.reshape(B, C, H * W)
        gf = get_graph_feature(fl, k=self.k)
        ef = self.ccc(fl, gf)
        return (fl + ef).reshape(B, C, H, W)

    def forward(self, frames):
        """frames: (B, T, 1, 128, 128)  →  (B, 128, 16, 16, T-1)"""
        B, T, C, H, W = frames.shape
        diffs = []
        for t in range(T - 1):
            pixel_diff = frames[:, t + 1] - frames[:, t]  # ← diff in PIXEL space
            diffs.append(self._encode(pixel_diff))
        return torch.stack(diffs, dim=-1)   # (B, 128, 16, 16, T-1)


class MOLBackbone_STATIC(nn.Module):
    """
    ABLATION D: Static single frame — no temporal information.
    Encodes only T=1 frame (the middle frame of the clip).
    This is the appearance oracle: if appearance alone is informative,
    AUC should be high. If it is near 0.5, appearance is not the signal.
    """
    def __init__(self, neighbor_k=4):
        super().__init__()
        self.conv_stack = ConvStack()
        self.fcc = FCC(dim=64, meta_kernel_size=32)
        self.ccc = CCC()
        self.k = neighbor_k

    def _frame_feat(self, frame):
        f = self.conv_stack(frame)
        f = self.fcc(f)
        B, C, H, W = f.shape
        fl = f.reshape(B, C, H * W)
        gf = get_graph_feature(fl, k=self.k)
        ef = self.ccc(fl, gf)
        return (fl + ef).reshape(B, C, H, W)

    def forward(self, frames):
        """
        frames: (B, T, 1, 128, 128)
        Uses ONLY the middle frame → (B, 128, 16, 16, 1)
        TemporalIdentityHead collapses temporal dim → works with T-1=1.
        """
        B, T, C, H, W = frames.shape
        mid = T // 2
        feat = self._frame_feat(frames[:, mid])        # single frame
        return feat.unsqueeze(-1)                       # (B, 128, 16, 16, 1)


###############################################################################
#  PART 2 — UNIFIED ABLATION MODEL
###############################################################################

class AblationModel(nn.Module):
    """
    Unified model wrapper for all four ablation conditions.
    Backbone varies; TemporalIdentityHead is shared architecture
    (weights are NOT shared across conditions — each is trained
    independently, but the head architecture is identical to ensure
    a fair comparison).
    """
    def __init__(self, condition: str, embed_dim=256, neighbor_k=4):
        super().__init__()
        assert condition in ALL_CONDITIONS, \
            f"Unknown condition '{condition}'. Choose from {ALL_CONDITIONS}"
        self.condition = condition

        if condition == "DIFF":
            self.backbone = MOLBackbone_DIFF(neighbor_k)
        elif condition == "RAW_FEAT":
            self.backbone = MOLBackbone_RAW_FEAT(neighbor_k)
        elif condition == "PIXEL_DIFF":
            self.backbone = MOLBackbone_PIXEL_DIFF(neighbor_k)
        elif condition == "STATIC":
            self.backbone = MOLBackbone_STATIC(neighbor_k)

        self.head = TemporalIdentityHead(embed_dim=embed_dim)

    def forward(self, frames):
        """frames: (B, T, 1, 128, 128)  →  (B, embed_dim) L2-normalised"""
        feat_volume = self.backbone(frames)   # (B, 128, 16, 16, T')
        return self.head(feat_volume)


###############################################################################
#  PART 3 — TRAINING
###############################################################################

def train_condition(args, condition: str) -> str:
    """Train one ablation condition and return checkpoint path."""
    ckpt_dir = Path(args.output_dir) / condition
    ckpt_path = ckpt_dir / "best_model.pt"

    if ckpt_path.exists() and not getattr(args, "retrain", False):
        logger.info(f"[{condition}] Checkpoint exists — skipping: {ckpt_path}")
        return str(ckpt_path)

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path(args.nvfair_split_root)

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds = NVFAIRFrameDataset(
        str(root / "train_files.csv"), args.frames_root,
        clip_length=args.clip_length, split="train",
    )
    val_ds = NVFAIRFrameDataset(
        str(root / "val_files.csv"), args.frames_root,
        clip_length=args.clip_length, split="val",
    )

    sampler = IdentityBatchSampler(
        train_ds,
        num_identities=args.num_ids,
        videos_per_identity=args.vids_per_id,
        steps=args.steps_per_epoch,
    )
    train_loader = DataLoader(
        train_ds, batch_sampler=sampler,
        num_workers=4, collate_fn=collate_fn, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=16, shuffle=False,
        num_workers=4, collate_fn=collate_fn, pin_memory=True,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = AblationModel(
        condition=condition,
        embed_dim=args.embed_dim,
        neighbor_k=args.neighbor_k,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"[{condition}] {n_params:,} params  device={device}")
    logger.info(f"[{condition}] {CONDITION_DESCRIPTIONS[condition]}")

    # ── Optimisation ──────────────────────────────────────────────────────────
    loss_fn = SupConLoss(temperature=args.temperature)
    optim = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.wd
    )
    total_steps = args.num_epochs * args.steps_per_epoch
    warmup_steps = 3 * args.steps_per_epoch

    def lr_fn(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_fn)
    scaler = GradScaler()

    best_auc = 0.0
    history = []

    config = dict(
        condition=condition,
        description=CONDITION_DESCRIPTIONS[condition],
        num_epochs=args.num_epochs,
        steps_per_epoch=args.steps_per_epoch,
        clip_length=args.clip_length,
        embed_dim=args.embed_dim,
        neighbor_k=args.neighbor_k,
        lr=args.lr, wd=args.wd,
        temperature=args.temperature,
        seed=args.seed,
    )
    with open(ckpt_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    logger.info(f"[{condition}] Training for {args.num_epochs} epochs …")

    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0.0

        for batch in tqdm(
            train_loader,
            desc=f"[{condition}] Epoch {epoch+1}/{args.num_epochs}",
            leave=False,
        ):
            frames = batch["frames"].to(device, non_blocking=True)
            labels = batch["driving_identity_idxs"].to(device, non_blocking=True)

            optim.zero_grad()
            with autocast():
                emb = model(frames)
                loss = loss_fn(emb, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()
            sched.step()
            total_loss += loss.item()

        avg_loss = total_loss / args.steps_per_epoch

        # Validate every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            vd = extract_all_embeddings(model, val_loader, device)
            auc, _ = compute_auc_paper_method(
                vd["embeddings"], vd["target_ids"], vd["is_self"]
            )
            logger.info(
                f"[{condition}] Epoch {epoch+1:3d}  "
                f"loss={avg_loss:.4f}  val_AUC={auc:.4f}"
            )
            history.append({"epoch": epoch + 1, "loss": avg_loss, "val_auc": auc})

            if auc > best_auc:
                best_auc = auc
                torch.save(
                    {
                        "model": model.state_dict(),
                        "auc": auc,
                        "epoch": epoch,
                        "condition": condition,
                        "config": config,
                    },
                    ckpt_path,
                )
                logger.info(f"[{condition}]   ★ New best val_AUC={auc:.4f}")
        else:
            logger.info(f"[{condition}] Epoch {epoch+1:3d}  loss={avg_loss:.4f}")
            history.append({"epoch": epoch + 1, "loss": avg_loss, "val_auc": None})

    with open(ckpt_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    logger.info(f"[{condition}] Done. Best val_AUC={best_auc:.4f}")
    return str(ckpt_path)


###############################################################################
#  PART 4 — EVALUATION
###############################################################################

def evaluate_condition(checkpoint_path: str, args, condition: str) -> dict:
    """Evaluate a trained checkpoint on the full test split."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})

    model = AblationModel(
        condition=condition,
        embed_dim=cfg.get("embed_dim", args.embed_dim),
        neighbor_k=cfg.get("neighbor_k", args.neighbor_k),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    root = Path(args.nvfair_split_root)
    test_ds = NVFAIRFrameDataset(
        str(root / "test_files.csv"), args.frames_root,
        clip_length=cfg.get("clip_length", args.clip_length),
        split="test",
    )
    test_loader = DataLoader(
        test_ds, batch_size=16, shuffle=False,
        num_workers=4, collate_fn=collate_fn, pin_memory=True,
    )

    logger.info(f"[eval/{condition}] {len(test_ds)} test videos …")
    td = extract_all_embeddings(model, test_loader, device)

    overall_auc, per_target = compute_auc_paper_method(
        td["embeddings"], td["target_ids"], td["is_self"]
    )

    # Per-generator AUC
    per_gen = {}
    for gen in ["facevid2vid", "tps", "lia"]:
        mask = torch.tensor([g == gen for g in td["generators"]])
        idx = torch.where(mask)[0]
        if len(idx) < 10:
            continue
        g_auc, _ = compute_auc_paper_method(
            td["embeddings"][idx],
            [td["target_ids"][i] for i in idx.tolist()],
            td["is_self"][idx],
        )
        per_gen[gen] = float(g_auc)

    # ROC curve for overall
    emb = F.normalize(td["embeddings"], p=2, dim=1)
    all_scores, all_labels = [], []
    for target in sorted(set(td["target_ids"])):
        mask = torch.tensor([t == target for t in td["target_ids"]])
        idx = torch.where(mask)[0]
        if len(idx) < 2:
            continue
        e = emb[idx]
        sf = td["is_self"][idx]
        si = torch.where(sf)[0]
        ci = torch.where(~sf)[0]
        if len(si) < 2 or len(ci) < 1:
            continue
        for i in range(len(si)):
            for j in range(i + 1, len(si)):
                all_scores.append(-torch.norm(e[si[i]] - e[si[j]], p=2).item())
                all_labels.append(1)
        for s in si:
            for c in ci:
                all_scores.append(-torch.norm(e[s] - e[c], p=2).item())
                all_labels.append(0)

    fpr, tpr, _ = roc_curve(all_labels, all_scores)

    result = {
        "condition": condition,
        "description": CONDITION_DESCRIPTIONS[condition],
        "overall_auc": float(overall_auc),
        "per_generator": per_gen,
        "n_targets": len(per_target),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "val_auc": float(ckpt.get("auc", 0.0)),
    }

    logger.info(
        f"[eval/{condition}] Test AUC={overall_auc:.4f}  "
        f"per_gen={per_gen}"
    )
    return result


###############################################################################
#  PART 5 — PLOTTING AND TABLE
###############################################################################

def plot_ablation_figure(results: Dict[str, dict], results_dir: str):
    """
    Two-panel figure:
      Left:  ROC curves for all 4 conditions (overall AUC)
      Right: Bar chart of AUC per generator per condition
    """
    results_dir = Path(results_dir)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Left panel: ROC curves ─────────────────────────────────────────────────
    ax = axes[0]
    condition_order = [c for c in ALL_CONDITIONS if c in results]
    for cond in condition_order:
        r = results[cond]
        fpr = np.array(r["fpr"])
        tpr = np.array(r["tpr"])
        auc = r["overall_auc"]
        linestyle = "-" if cond == "DIFF" else "--"
        lw = 2.5 if cond == "DIFF" else 1.8
        ax.plot(
            fpr, tpr,
            color=CONDITION_COLORS[cond],
            linewidth=lw,
            linestyle=linestyle,
            label=f"{cond}  (AUC={auc:.3f})",
            zorder=3 if cond == "DIFF" else 2,
        )

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, lw=0.8)
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curves — Ablation Study\n(Dynamic vs Appearance Features)",
                 fontsize=11, fontweight="bold")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)

    # Annotation box explaining interpretation
    ax.text(
        0.98, 0.35,
        "DIFF >> STATIC\n→ dynamics matter\n\nDIFF ≥ RAW_FEAT\n→ appearance doesn't help",
        transform=ax.transAxes,
        fontsize=7.5, va="top", ha="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                  edgecolor="grey", alpha=0.8),
    )

    # ── Right panel: Per-generator bar chart ──────────────────────────────────
    ax2 = axes[1]
    generators = ["facevid2vid", "tps", "lia"]
    gen_labels = ["Face-vid2vid", "TPS", "LIA"]
    x = np.arange(len(generators))
    width = 0.18
    n_conds = len(condition_order)
    offsets = np.linspace(-(n_conds - 1) * width / 2,
                          (n_conds - 1) * width / 2, n_conds)

    for i, cond in enumerate(condition_order):
        r = results[cond]
        pg = r.get("per_generator", {})
        aucs = [pg.get(g, 0.0) for g in generators]
        hatch = "" if cond == "DIFF" else "///"
        bars = ax2.bar(
            x + offsets[i], aucs,
            width=width,
            label=cond,
            color=CONDITION_COLORS[cond],
            hatch=hatch,
            alpha=0.85,
            edgecolor="white" if cond == "DIFF" else CONDITION_COLORS[cond],
        )
        # Value labels on bars
        for bar, v in zip(bars, aucs):
            if v > 0.01:
                ax2.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.003,
                    f"{v:.3f}", ha="center", va="bottom",
                    fontsize=6.5, rotation=45,
                )

    ax2.set_xticks(x)
    ax2.set_xticklabels(gen_labels, fontsize=10)
    ax2.set_ylabel("AUC", fontsize=11)
    ax2.set_ylim(0.45, 1.0)
    ax2.set_title("Per-Generator AUC — Ablation Study", fontsize=11,
                  fontweight="bold")
    ax2.legend(loc="lower right", fontsize=8.5, framealpha=0.9)
    ax2.axhline(0.5, color="black", linestyle=":", alpha=0.4, linewidth=0.9,
                label="Random chance")
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    pdf_path = str(results_dir / "ablation_dynamic_vs_appearance.pdf")
    png_path = str(results_dir / "ablation_dynamic_vs_appearance.png")
    plt.savefig(pdf_path, bbox_inches="tight", dpi=150)
    plt.savefig(png_path, bbox_inches="tight", dpi=150)
    plt.close()
    logger.info(f"Figure → {pdf_path}")
    logger.info(f"Figure → {png_path}")
    return pdf_path, png_path


def plot_training_curves(results_dir: str):
    """
    Optional: plot validation AUC over training epochs for all conditions.
    Reveals convergence speed differences.
    """
    results_dir = Path(results_dir)
    fig, ax = plt.subplots(figsize=(8, 5))

    for cond in ALL_CONDITIONS:
        hist_path = results_dir / cond / "training_history.json"
        if not hist_path.exists():
            # Try output dir
            hist_path = results_dir.parent / "outputs" / "ablation_dynamic" / cond / "training_history.json"
        if not hist_path.exists():
            continue
        with open(hist_path) as f:
            hist = json.load(f)
        epochs = [h["epoch"] for h in hist if h["val_auc"] is not None]
        aucs   = [h["val_auc"] for h in hist if h["val_auc"] is not None]
        if not epochs:
            continue
        ls = "-" if cond == "DIFF" else "--"
        ax.plot(epochs, aucs, color=CONDITION_COLORS[cond],
                linewidth=2.0, linestyle=ls,
                label=f"{cond}: {CONDITION_DESCRIPTIONS[cond][:35]}…",
                marker="o", markersize=4)

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Validation AUC", fontsize=11)
    ax.set_title("Training Convergence — All Ablation Conditions", fontsize=11,
                 fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.axhline(0.5, color="black", linestyle=":", alpha=0.3)

    path = str(results_dir / "ablation_training_curves.png")
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    logger.info(f"Training curves → {path}")


def print_ablation_table(results: Dict[str, dict]):
    """Print the paper-ready comparison table."""
    div = "=" * 82
    print(f"\n{div}")
    print("  TABLE: Dynamic vs Appearance Features Ablation  (AUC — higher is better)")
    print(div)
    print(f"  {'Condition':<14} {'Description':<38} {'Overall':>8} "
          f"{'fv2v':>7} {'TPS':>7} {'LIA':>7}")
    print("-" * 82)

    for cond in ALL_CONDITIONS:
        if cond not in results:
            continue
        r = results[cond]
        pg = r.get("per_generator", {})
        desc = CONDITION_DESCRIPTIONS[cond][:37]
        marker = " ◄" if cond == "DIFF" else "  "
        print(
            f"  {cond:<14} {desc:<38} "
            f"{r['overall_auc']:>8.4f}"
            f"{pg.get('facevid2vid', 0.0):>7.4f}"
            f"{pg.get('tps', 0.0):>7.4f}"
            f"{pg.get('lia', 0.0):>7.4f}"
            f"{marker}"
        )

    print(div)
    if "DIFF" in results and "STATIC" in results:
        diff_auc = results["DIFF"]["overall_auc"]
        static_auc = results["STATIC"]["overall_auc"]
        delta = diff_auc - static_auc
        print(f"\n  DIFF vs STATIC: Δ AUC = +{delta:.4f}")
        print(f"  → Motion dynamics contribute {delta/max(diff_auc-0.5,1e-6)*100:.1f}% "
              f"of the performance above chance.")
    if "DIFF" in results and "RAW_FEAT" in results:
        diff_auc = results["DIFF"]["overall_auc"]
        raw_auc = results["RAW_FEAT"]["overall_auc"]
        delta = diff_auc - raw_auc
        sign = "+" if delta >= 0 else ""
        print(f"\n  DIFF vs RAW_FEAT: Δ AUC = {sign}{delta:.4f}")
        if delta >= 0:
            print(f"  → Subtracting appearance IMPROVES performance.")
            print(f"    Appearance features are a distractor, not a useful signal.")
        else:
            print(f"  → Note: RAW_FEAT outperforms DIFF by {-delta:.4f}.")
            print(f"    Investigate whether appearance leakage is helping or hurting.")
    print(div)


###############################################################################
#  MAIN
###############################################################################

def main():
    p = argparse.ArgumentParser(
        description="Ablation: dynamic features vs appearance features"
    )

    # Paths
    p.add_argument("--nvfair_split_root", default="NVFAIR_split")
    p.add_argument("--frames_root", default="nvfair_frames")
    p.add_argument("--output_dir", default="outputs/ablation_dynamic",
                   help="Checkpoint directory")
    p.add_argument("--results_dir", default="results/ablation_dynamic",
                   help="Results + figures directory")

    # Model (match your best run)
    p.add_argument("--clip_length", type=int, default=64)
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--neighbor_k", type=int, default=4)

    # Training
    p.add_argument("--num_epochs", type=int, default=30,
                   help="Ablation needs fewer epochs than full training")
    p.add_argument("--steps_per_epoch", type=int, default=200)
    p.add_argument("--num_ids", type=int, default=16)
    p.add_argument("--vids_per_id", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--seed", type=int, default=42)

    # Flow control
    p.add_argument("--conditions", nargs="+", default=ALL_CONDITIONS,
                   choices=ALL_CONDITIONS,
                   help="Which conditions to run (default: all four)")
    p.add_argument("--skip_training", action="store_true")
    p.add_argument("--skip_eval", action="store_true")
    p.add_argument("--plot_only", action="store_true")
    p.add_argument("--retrain", action="store_true")
    p.add_argument("--no_training_curves", action="store_true")

    args = p.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    all_results: Dict[str, dict] = {}

    # ── Load existing results if available (for plot_only) ────────────────────
    for cond in args.conditions:
        r_path = Path(args.results_dir) / f"result_{cond}.json"
        if r_path.exists():
            with open(r_path) as f:
                all_results[cond] = json.load(f)
            logger.info(f"[{cond}] Loaded existing result: AUC={all_results[cond]['overall_auc']:.4f}")

    if not args.plot_only:
        checkpoints: Dict[str, str] = {}

        # ── Step 1: Train ──────────────────────────────────────────────────────
        if not args.skip_training:
            logger.info("=" * 60)
            logger.info("STEP 1 — Training ablation conditions")
            logger.info("=" * 60)
            for cond in args.conditions:
                ckpt = train_condition(args, cond)
                checkpoints[cond] = ckpt
        else:
            for cond in args.conditions:
                ckpt = Path(args.output_dir) / cond / "best_model.pt"
                if ckpt.exists():
                    checkpoints[cond] = str(ckpt)
                    logger.info(f"[{cond}] Using checkpoint: {ckpt}")
                else:
                    logger.warning(f"[{cond}] Checkpoint NOT found: {ckpt}")

        # ── Step 2: Evaluate ───────────────────────────────────────────────────
        if not args.skip_eval:
            logger.info("=" * 60)
            logger.info("STEP 2 — Evaluating ablation conditions")
            logger.info("=" * 60)
            for cond, ckpt in checkpoints.items():
                result = evaluate_condition(ckpt, args, cond)
                all_results[cond] = result

                r_path = Path(args.results_dir) / f"result_{cond}.json"
                with open(r_path, "w") as f:
                    json.dump(result, f, indent=2, default=str)
                logger.info(f"[{cond}] Saved → {r_path}")

    # ── Step 3: Plot and table ─────────────────────────────────────────────────
    if all_results:
        logger.info("=" * 60)
        logger.info("STEP 3 — Plotting and printing table")
        logger.info("=" * 60)

        pdf, png = plot_ablation_figure(all_results, args.results_dir)
        print_ablation_table(all_results)

        if not args.no_training_curves:
            plot_training_curves(args.results_dir)

        # Save combined results JSON
        combined_path = Path(args.results_dir) / "ablation_all_results.json"
        with open(combined_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

        logger.info(f"\nDone!")
        logger.info(f"  PDF  → {pdf}")
        logger.info(f"  PNG  → {png}")
        logger.info(f"  JSON → {combined_path}")
    else:
        logger.warning("No results to plot. Run without --plot_only first.")


if __name__ == "__main__":
    main()