#!/usr/bin/env python3
"""
==========================================================================
Generalization Experiment for PIXEL_DIFF Model
==========================================================================

This script runs the cross-generator generalization experiment (NVFAIR Fig. 6)
using the PIXEL_DIFF model instead of the standard MOL DIFF model.

PIXEL_DIFF computes differences in PIXEL space before encoding:
  diff_t = frame_{t+1} - frame_t  (in pixel space)
  Then: diff_t → ConvStack → FCC → CCC → features

This contrasts with DIFF which computes differences in FEATURE space:
  feat_t = ConvStack → FCC → CCC (frame_t)
  feat_{t+1} = ConvStack → FCC → CCC (frame_{t+1})
  diff_t = feat_{t+1} - feat_t

WORKFLOW
--------
  Step 1  Train one PIXEL_DIFF model per generator (facevid2vid / tps / lia)
  Step 2  Evaluate every checkpoint on ALL three generators
  Step 3  Plot 1 × 3 ROC figure and comparison table

USAGE
-----
  CUDA_VISIBLE_DEVICES=0 python run_generalization_pixel_diff.py \
      --nvfair_split_root NVFAIR_split \
      --frames_root       nvfair_frames \
      --output_dir        outputs/gen_pixel_diff \
      --results_dir       results/gen_pixel_diff \
      --num_epochs 50 \
      --steps_per_epoch 200 \
      --clip_length 64
"""

import argparse
import json
import logging
import math
import os
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm

try:
    from sklearn.metrics import roc_auc_score, roc_curve
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as e:
    sys.exit(f"[ERROR] Missing dependency: {e}")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Constants
GENERATORS = ["facevid2vid", "tps", "lia"]
GEN_LABELS = {"facevid2vid": "Face-vid2vid", "tps": "TPS", "lia": "LIA"}
COLORS = {"facevid2vid": "#E8624A", "tps": "#4A8FCC", "lia": "#5CB85C"}

# Reference AUC values from NVFAIR paper Fig. 6
PAPER_AUCS = {
    "facevid2vid": {"facevid2vid": 0.87, "tps": 0.82, "lia": 0.84},
    "tps":         {"facevid2vid": 0.85, "tps": 0.85, "lia": 0.84},
    "lia":         {"facevid2vid": 0.83, "tps": 0.82, "lia": 0.84},
}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


###############################################################################
#  MOL BACKBONE MODULES (from train_mol_avatar.py)
###############################################################################

class ConvStack(nn.Module):
    """Initial conv encoder: 1×128×128 → 128×16×16"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, 4, stride=2),
            nn.BatchNorm2d(8), nn.ReLU(inplace=True),
            nn.Conv2d(8, 32, 3, stride=2),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 2, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class FCC(nn.Module):
    """Fully-Connected Convolution (transformer-style)"""
    def __init__(self, dim=64, meta_kernel_size=32, use_pe=True, bias=True):
        super().__init__()
        self.dim = dim
        self.use_pe = use_pe

        self.pre_norm_1 = nn.BatchNorm2d(dim)
        self.pre_norm_2 = nn.BatchNorm2d(dim)

        self.mk_1_H = nn.Conv2d(dim, dim, (meta_kernel_size, 1), groups=dim).weight
        self.mk_1_W = nn.Conv2d(dim, dim, (1, meta_kernel_size), groups=dim).weight
        self.mk_2_H = nn.Conv2d(dim, dim, (meta_kernel_size, 1), groups=dim).weight
        self.mk_2_W = nn.Conv2d(dim, dim, (1, meta_kernel_size), groups=dim).weight

        if bias:
            self.b1H = nn.Parameter(torch.randn(dim))
            self.b1W = nn.Parameter(torch.randn(dim))
            self.b2H = nn.Parameter(torch.randn(dim))
            self.b2W = nn.Parameter(torch.randn(dim))
        else:
            self.b1H = self.b1W = self.b2H = self.b2W = None

        if use_pe:
            self.pe_1H = nn.Parameter(torch.randn(1, dim, meta_kernel_size, 1))
            self.pe_1W = nn.Parameter(torch.randn(1, dim, 1, meta_kernel_size))
            self.pe_2H = nn.Parameter(torch.randn(1, dim, meta_kernel_size, 1))
            self.pe_2W = nn.Parameter(torch.randn(1, dim, 1, meta_kernel_size))

    def _k(self, s):
        return (self.mk_1_H[:, :, :s, :], self.mk_1_W[:, :, :, :s],
                self.mk_2_H[:, :, :s, :], self.mk_2_W[:, :, :, :s])

    def _pe(self, s):
        return (self.pe_1H[:, :, :s, :].expand(1, self.dim, s, s),
                self.pe_1W[:, :, :, :s].expand(1, self.dim, s, s),
                self.pe_2H[:, :, :s, :].expand(1, self.dim, s, s),
                self.pe_2W[:, :, :, :s].expand(1, self.dim, s, s))

    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=1)
        x1r, x2r = x1, x2
        _, _, s, _ = x1.shape

        K1H, K1W, K2H, K2W = self._k(s)

        if self.use_pe:
            p1H, p1W, p2H, p2W = self._pe(s)
            x1, x2 = x1 + p1H, x2 + p1W

        x1, x2 = self.pre_norm_1(x1), self.pre_norm_2(x2)

        x1 = F.conv2d(torch.cat((x1, x1[:, :, :-1, :]), 2),
                       weight=K1H, bias=self.b1H, groups=self.dim)
        x2 = F.conv2d(torch.cat((x2, x2[:, :, :, :-1]), 3),
                       weight=K1W, bias=self.b1W, groups=self.dim)

        if self.use_pe:
            x1, x2 = x1 + p2W, x2 + p2H

        x1 = F.conv2d(torch.cat((x1, x1[:, :, :, :-1]), 3),
                       weight=K2W, bias=self.b2W, groups=self.dim)
        x2 = F.conv2d(torch.cat((x2, x2[:, :, :-1, :]), 2),
                       weight=K2H, bias=self.b2H, groups=self.dim)

        return torch.cat((x1r + x1, x2r + x2), dim=1)


def knn_cos_sim(x, k):
    x_n = F.normalize(x, p=2, dim=1)
    sim = torch.bmm(x_n.transpose(1, 2), x_n)
    return sim.topk(k=k, dim=-1)[1]


def get_graph_feature(x, k=4):
    B, C, N = x.size()
    idx = knn_cos_sim(x, k)
    device = x.device
    base = torch.arange(0, B, device=device).view(-1, 1, 1) * N
    idx = (idx + base).view(-1)

    xt = x.transpose(1, 2).contiguous()
    feat = xt.view(B * N, -1)[idx].view(B, N, k, C)
    xr = xt.view(B, N, 1, C).expand_as(feat)
    return (feat - xr).permute(0, 3, 1, 2)


class CCC(nn.Module):
    """Channel Correspondence Convolution (graph-style)"""
    def __init__(self):
        super().__init__()
        self.edge_fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
        )

    def forward(self, f_i, f_j_k):
        edge_agg = f_j_k.max(dim=-1)[0]
        return F.relu(f_i + edge_agg, inplace=False)


class TemporalIdentityHead(nn.Module):
    """3D-conv temporal aggregation → 256-d identity embedding"""
    def __init__(self, embed_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(128, 64, kernel_size=(3, 3, 3),
                      stride=(1, 2, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(64), nn.ReLU(inplace=True),
            nn.Conv3d(64, 32, kernel_size=(3, 3, 3),
                      stride=(1, 2, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(32), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((4, 4, 1)),
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, embed_dim),
        )

    def forward(self, diff_feats):
        x = self.net(diff_feats)
        x = x.squeeze(-1).flatten(1)
        return F.normalize(self.fc(x), p=2, dim=-1)


###############################################################################
#  PIXEL_DIFF BACKBONE — The key difference from standard MOL
###############################################################################

class MOLBackbone_PIXEL_DIFF(nn.Module):
    """
    PIXEL_DIFF: Computes differences in PIXEL space, THEN encodes.
    
    This is the key difference from the standard DIFF model:
    - Standard DIFF: encode each frame, then subtract features
    - PIXEL_DIFF: subtract pixels first, then encode the difference
    
    Pipeline:
      pixel_diff_t = frame_{t+1} - frame_t  (in pixel space)
      Then: pixel_diff_t → ConvStack → FCC → CCC → encoded_diff
    """
    def __init__(self, neighbor_k=4):
        super().__init__()
        self.conv_stack = ConvStack()
        self.fcc = FCC(dim=64, meta_kernel_size=32)
        self.ccc = CCC()
        self.k = neighbor_k

    def _encode(self, pixel_diff):
        """Encode a pixel-level difference: (B, 1, 128, 128) → (B, 128, 16, 16)"""
        f = self.conv_stack(pixel_diff)
        f = self.fcc(f)
        B, C, H, W = f.shape
        fl = f.reshape(B, C, H * W)
        gf = get_graph_feature(fl, k=self.k)
        ef = self.ccc(fl, gf)
        return (fl + ef).reshape(B, C, H, W)

    def forward(self, frames):
        """
        frames: (B, T, 1, 128, 128)
        Returns: (B, 128, 16, 16, T-1)
        """
        B, T, C, H, W = frames.shape
        diffs = []
        for t in range(T - 1):
            # KEY DIFFERENCE: Subtract in PIXEL space first
            pixel_diff = frames[:, t + 1] - frames[:, t]
            # Then encode the pixel difference
            diffs.append(self._encode(pixel_diff))
        return torch.stack(diffs, dim=-1)


class PixelDiffModel(nn.Module):
    """Complete PIXEL_DIFF model: raw video → identity embedding"""
    def __init__(self, embed_dim=256, neighbor_k=4):
        super().__init__()
        self.backbone = MOLBackbone_PIXEL_DIFF(neighbor_k=neighbor_k)
        self.head = TemporalIdentityHead(embed_dim=embed_dim)

    def forward(self, frames):
        """
        frames: (B, T, 1, 128, 128) — grayscale video clip
        Returns: (B, embed_dim) — L2-normalised embedding
        """
        diff_feats = self.backbone(frames)
        return self.head(diff_feats)


###############################################################################
#  DATASET AND DATALOADER
###############################################################################

class NVFAIRFrameDataset(Dataset):
    """Loads pre-extracted 128×128 grayscale frames for each NVFAIR video."""
    def __init__(self, csv_path, frames_root, clip_length=64,
                 split='train', generator_filter=None):
        self.frames_root = Path(frames_root)
        self.clip_length = clip_length
        self.split = split

        df = pd.read_csv(csv_path,
                         dtype={'target_identity': str, 'driving_identity': str},
                         low_memory=False)
        if generator_filter:
            df = df[df['generator'] == generator_filter]

        self.videos = []
        self.identity_to_videos = defaultdict(lambda: {'self': [], 'cross': []})

        all_drivers = sorted(df['driving_identity'].unique())
        self.driver_to_idx = {str(d): i for i, d in enumerate(all_drivers)}

        skipped = 0
        for _, row in df.iterrows():
            new_path = row['new_path']
            frame_dir = self.frames_root / new_path.replace('.mp4', '')

            if not frame_dir.exists():
                skipped += 1
                continue

            n_frames = len(list(frame_dir.glob('*.png')))
            if n_frames < 2:
                skipped += 1
                continue

            driver = str(row['driving_identity'])
            target = str(row['target_identity'])
            if 'is_self_reenactment' in row:
                is_self = bool(row['is_self_reenactment'])
            else:
                is_self = (driver == target)

            self.videos.append({
                'frame_dir': str(frame_dir),
                'num_frames': n_frames,
                'driving_identity': driver,
                'driving_identity_idx': self.driver_to_idx.get(driver, 0),
                'target_identity': target,
                'is_self_reenactment': is_self,
                'generator': row.get('generator', 'unknown'),
            })

            key = 'self' if is_self else 'cross'
            self.identity_to_videos[driver][key].append(len(self.videos) - 1)

        if skipped:
            logger.warning(f"Skipped {skipped} videos (no frames found)")
        logger.info(f"NVFAIRFrameDataset [{split}]: {len(self.videos)} videos, "
                    f"{len(self.driver_to_idx)} identities")

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        v = self.videos[idx]
        frame_dir = Path(v['frame_dir'])
        files = sorted(frame_dir.glob('*.png'))
        n = len(files)

        if self.split == 'train' and n > self.clip_length:
            start = random.randint(0, n - self.clip_length)
        elif n > self.clip_length:
            start = (n - self.clip_length) // 2
        else:
            start = 0

        sel = files[start:start + self.clip_length]

        imgs = []
        for fp in sel:
            im = cv2.imread(str(fp), cv2.IMREAD_GRAYSCALE)
            if im is None:
                im = np.zeros((128, 128), dtype=np.uint8)
            imgs.append(im.astype(np.float32) / 255.0)

        while len(imgs) < self.clip_length:
            imgs.append(imgs[-1])

        frames = np.stack(imgs)[:, None, :, :]
        frames = torch.from_numpy(frames)
        frames = (frames - frames.mean()) / (frames.std() + 1e-6)

        return {
            'frames': frames,
            'length': min(n, self.clip_length),
            'driving_identity': v['driving_identity'],
            'driving_identity_idx': v['driving_identity_idx'],
            'target_identity': v['target_identity'],
            'is_self_reenactment': v['is_self_reenactment'],
            'generator': v['generator'],
        }


def collate_fn(batch):
    return {
        'frames': torch.stack([b['frames'] for b in batch]),
        'lengths': torch.tensor([b['length'] for b in batch]),
        'driving_identities': [b['driving_identity'] for b in batch],
        'driving_identity_idxs': torch.tensor([b['driving_identity_idx'] for b in batch]),
        'target_identities': [b['target_identity'] for b in batch],
        'is_self_reenactment': torch.tensor([b['is_self_reenactment'] for b in batch]),
        'generators': [b['generator'] for b in batch],
    }


class IdentityBatchSampler(Sampler):
    """Each batch: N identities × M videos/identity."""
    def __init__(self, dataset, num_identities=16, videos_per_identity=8, steps=200):
        self.ds = dataset
        self.n_ids = num_identities
        self.v_per_id = videos_per_identity
        self.steps = steps

        self.valid_ids = [
            ident for ident, vids in dataset.identity_to_videos.items()
            if len(vids['self']) + len(vids['cross']) >= videos_per_identity
        ]
        logger.info(f"BatchSampler: {len(self.valid_ids)} valid identities "
                    f"(need ≥{videos_per_identity} videos)")

    def __iter__(self):
        for _ in range(self.steps):
            ids = random.sample(self.valid_ids, min(self.n_ids, len(self.valid_ids)))
            batch = []
            for ident in ids:
                pool = (self.ds.identity_to_videos[ident]['self'] +
                        self.ds.identity_to_videos[ident]['cross'])
                if len(pool) >= self.v_per_id:
                    batch.extend(random.sample(pool, self.v_per_id))
                else:
                    batch.extend(random.choices(pool, k=self.v_per_id))
            yield batch

    def __len__(self):
        return self.steps


###############################################################################
#  LOSS AND EVALUATION
###############################################################################

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.t = temperature

    def forward(self, emb, labels):
        B = emb.shape[0]
        device = emb.device
        emb = F.normalize(emb, p=2, dim=1)

        sim = emb @ emb.T / self.t
        eye = torch.eye(B, device=device)
        pos = (labels.view(-1, 1) == labels.view(1, -1)).float() - eye

        sim = sim - sim.max(dim=1, keepdim=True)[0].detach()
        exp = torch.exp(sim) * (1 - eye)
        log_p = sim - torch.log(exp.sum(1, keepdim=True) + 1e-8)

        n_pos = pos.sum(1).clamp(min=1)
        return -(pos * log_p).sum(1).div(n_pos).mean()


def compute_auc_paper_method(embeddings, target_ids, is_self):
    """Paper-exact evaluation (Equation 6)"""
    embeddings = F.normalize(embeddings, p=2, dim=1)
    per_target = {}

    for target in sorted(set(target_ids)):
        mask = torch.tensor([t == target for t in target_ids])
        idx = torch.where(mask)[0]
        if len(idx) < 2:
            continue

        e = embeddings[idx]
        sf = is_self[idx]
        si = torch.where(sf)[0]
        ci = torch.where(~sf)[0]

        if len(si) < 2 or len(ci) < 1:
            continue

        scores, labels = [], []
        for i in range(len(si)):
            for j in range(i + 1, len(si)):
                scores.append(-torch.norm(e[si[i]] - e[si[j]], p=2).item())
                labels.append(1)
        for s in si:
            for c in ci:
                scores.append(-torch.norm(e[s] - e[c], p=2).item())
                labels.append(0)

        if len(set(labels)) < 2:
            continue
        try:
            per_target[target] = roc_auc_score(labels, scores)
        except Exception:
            pass

    overall = np.mean(list(per_target.values())) if per_target else 0.5
    return overall, per_target


def extract_all_embeddings(model, loader, device):
    model.eval()
    out = {'emb': [], 'tgt': [], 'sf': [], 'gen': [], 'drv': []}
    with torch.no_grad():
        for batch in tqdm(loader, desc="Embed", leave=False):
            e = model(batch['frames'].to(device))
            out['emb'].append(e.cpu())
            out['tgt'].extend(batch['target_identities'])
            out['sf'].append(batch['is_self_reenactment'])
            out['gen'].extend(batch['generators'])
            out['drv'].extend(batch['driving_identities'])
    return {
        'embeddings': torch.cat(out['emb']),
        'target_ids': out['tgt'],
        'is_self': torch.cat(out['sf']).bool(),
        'generators': out['gen'],
        'driving_ids': out['drv'],
    }


###############################################################################
#  TRAINING
###############################################################################

def train_on_generator(args, train_gen: str) -> str:
    """Train a PixelDiffModel on videos from a single generator."""
    ckpt_dir = Path(args.output_dir) / f"gen_{train_gen}"
    ckpt_path = ckpt_dir / "best_model.pt"

    if ckpt_path.exists() and not getattr(args, "retrain", False):
        logger.info(f"[{train_gen}] Checkpoint exists — skipping: {ckpt_path}")
        return str(ckpt_path)

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    root = Path(args.nvfair_split_root)

    # Datasets
    train_ds = NVFAIRFrameDataset(
        str(root / "train_files.csv"), args.frames_root,
        clip_length=args.clip_length, split="train",
        generator_filter=train_gen,
    )
    val_ds = NVFAIRFrameDataset(
        str(root / "val_files.csv"), args.frames_root,
        clip_length=args.clip_length, split="val",
        generator_filter=train_gen,
    )

    if len(train_ds) == 0:
        raise RuntimeError(f"No training videos found for generator '{train_gen}'.")

    sampler = IdentityBatchSampler(
        train_ds, num_identities=args.num_ids,
        videos_per_identity=args.vids_per_id, steps=args.steps_per_epoch,
    )
    train_loader = DataLoader(
        train_ds, batch_sampler=sampler,
        num_workers=4, collate_fn=collate_fn, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=16, shuffle=False,
        num_workers=4, collate_fn=collate_fn, pin_memory=True,
    )

    # Model - PIXEL_DIFF
    model = PixelDiffModel(
        embed_dim=args.embed_dim,
        neighbor_k=args.neighbor_k,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"[{train_gen}] PIXEL_DIFF Model: {n_params:,} params  device={device}")

    # Optimization
    loss_fn = SupConLoss(temperature=args.temperature)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    
    total_steps = args.num_epochs * args.steps_per_epoch
    warmup_steps = 5 * args.steps_per_epoch

    def lr_fn(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_fn)
    scaler = GradScaler()

    best_auc = 0.0
    config = dict(
        model_type="PIXEL_DIFF",
        train_generator=train_gen,
        num_epochs=args.num_epochs,
        steps_per_epoch=args.steps_per_epoch,
        clip_length=args.clip_length,
        embed_dim=args.embed_dim,
        neighbor_k=args.neighbor_k,
        lr=args.lr, wd=args.wd,
        temperature=args.temperature,
        num_ids=args.num_ids,
        vids_per_id=args.vids_per_id,
        seed=args.seed,
    )
    with open(ckpt_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    logger.info(f"[{train_gen}] Starting PIXEL_DIFF training  epochs={args.num_epochs}")

    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0.0

        for batch in tqdm(
            train_loader,
            desc=f"[{train_gen}] Epoch {epoch+1}/{args.num_epochs}",
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

        # Validation
        if (epoch + 1) % 5 == 0 or epoch == 0:
            vd = extract_all_embeddings(model, val_loader, device)
            auc, _ = compute_auc_paper_method(
                vd["embeddings"], vd["target_ids"], vd["is_self"]
            )
            logger.info(
                f"[{train_gen}] Epoch {epoch+1:3d}  loss={avg_loss:.4f}  val_AUC={auc:.4f}"
            )
            if auc > best_auc:
                best_auc = auc
                torch.save(
                    {"model": model.state_dict(), "auc": auc, "epoch": epoch, "config": config},
                    ckpt_path,
                )
                logger.info(f"[{train_gen}]   ★ New best val_AUC={auc:.4f}")
        else:
            logger.info(f"[{train_gen}] Epoch {epoch+1:3d}  loss={avg_loss:.4f}")

    logger.info(f"[{train_gen}] Training done. Best val_AUC={best_auc:.4f}")
    return str(ckpt_path)


###############################################################################
#  EVALUATION
###############################################################################

def _roc_for_generator(embeddings, target_ids, is_self):
    """Compute ROC curve data and AUC."""
    embeddings = F.normalize(embeddings, p=2, dim=1)
    all_scores, all_labels = [], []

    for target in sorted(set(target_ids)):
        mask = torch.tensor([t == target for t in target_ids])
        idx = torch.where(mask)[0]
        if len(idx) < 2:
            continue

        e = embeddings[idx]
        sf = is_self[idx]
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

    if len(set(all_labels)) < 2 or len(all_labels) < 4:
        return None

    try:
        auc = roc_auc_score(all_labels, all_scores)
        fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
        return {
            "auc": float(auc),
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "n_pairs": len(all_labels),
        }
    except Exception as exc:
        logger.warning(f"ROC computation failed: {exc}")
        return None


def evaluate_checkpoint_cross_generator(checkpoint_path: str, args, train_gen: str) -> dict:
    """Evaluate on ALL generators."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"[eval] Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})

    model = PixelDiffModel(
        embed_dim=cfg.get("embed_dim", args.embed_dim),
        neighbor_k=cfg.get("neighbor_k", args.neighbor_k),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    root = Path(args.nvfair_split_root)
    test_ds = NVFAIRFrameDataset(
        str(root / "test_files.csv"), args.frames_root,
        clip_length=cfg.get("clip_length", args.clip_length),
        split="test", generator_filter=None,
    )
    test_loader = DataLoader(
        test_ds, batch_size=16, shuffle=False,
        num_workers=4, collate_fn=collate_fn, pin_memory=True,
    )

    logger.info(f"[eval/{train_gen}] Extracting embeddings for {len(test_ds)} videos …")
    td = extract_all_embeddings(model, test_loader, device)

    emb, tgt, sf, gen = td["embeddings"], td["target_ids"], td["is_self"], td["generators"]

    # Per-generator ROC
    per_gen = {}
    for test_gen in GENERATORS:
        mask = torch.tensor([g == test_gen for g in gen])
        idx = torch.where(mask)[0]
        if len(idx) < 10:
            continue
        roc = _roc_for_generator(emb[idx], [tgt[i] for i in idx.tolist()], sf[idx])
        if roc is not None:
            per_gen[test_gen] = roc
            logger.info(f"[eval/{train_gen}→{test_gen}] AUC={roc['auc']:.4f}")

    overall_auc, _ = compute_auc_paper_method(emb, tgt, sf)
    logger.info(f"[eval/{train_gen}] Overall AUC={overall_auc:.4f}")

    return {
        "train_generator": train_gen,
        "per_generator": per_gen,
        "overall_auc": float(overall_auc),
        "val_auc": float(ckpt.get("auc", 0.0)),
    }


###############################################################################
#  PLOTTING
###############################################################################

def plot_generalization_figure(results_dir: str, output_path: str = None):
    """Draw a 1 × 3 ROC panel matching NVFAIR Fig. 6."""
    results_dir = Path(results_dir)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle(
        "PIXEL_DIFF Generalization to New Generators\n"
        "(Pixel-level differencing vs Original NVFAIR paper — FLAME features)",
        fontsize=12, fontweight="bold", y=1.02,
    )

    for col, train_gen in enumerate(GENERATORS):
        ax = axes[col]
        roc_file = results_dir / f"roc_data_{train_gen}.json"

        if not roc_file.exists():
            ax.text(0.5, 0.5, f"No results\nfor {GEN_LABELS[train_gen]}",
                    ha="center", va="center", transform=ax.transAxes, fontsize=10)
            ax.set_title(f"Trained on {GEN_LABELS[train_gen]}", fontsize=11)
            _style_ax(ax, col)
            continue

        with open(roc_file) as f:
            roc_data = json.load(f)

        for test_gen in GENERATORS:
            if test_gen not in roc_data:
                continue
            d = roc_data[test_gen]
            auc = d["auc"]
            fpr = np.array(d["fpr"])
            tpr = np.array(d["tpr"])
            ax.plot(fpr, tpr, color=COLORS[test_gen], linewidth=2.2,
                    label=f"{GEN_LABELS[test_gen]} – {auc:.2f}")

        # Paper reference (dashed)
        if train_gen in PAPER_AUCS:
            for test_gen in GENERATORS:
                p_auc = PAPER_AUCS[train_gen].get(test_gen)
                if p_auc is None:
                    continue
                ax.plot([], [], color=COLORS[test_gen], linewidth=1.5,
                        linestyle="--", alpha=0.55,
                        label=f"{GEN_LABELS[test_gen]} (paper) – {p_auc:.2f}")

        ax.set_title(f"Trained on {GEN_LABELS[train_gen]}", fontsize=11)
        _style_ax(ax, col)

        handles, labels_ = ax.get_legend_handles_labels()
        solid = [(h, l) for h, l in zip(handles, labels_) if "(paper)" not in l]
        dashed = [(h, l) for h, l in zip(handles, labels_) if "(paper)" in l]
        all_h = [h for h, _ in solid] + [h for h, _ in dashed]
        all_l = [l for _, l in solid] + [l for _, l in dashed]
        ax.legend(all_h, all_l, loc="lower right", fontsize=7.5, framealpha=0.9)

    plt.tight_layout()

    if output_path is None:
        output_path = str(results_dir / "fig6_generalization_pixel_diff.pdf")

    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    png_path = output_path.replace(".pdf", ".png")
    plt.savefig(png_path, bbox_inches="tight", dpi=150)
    plt.close()

    logger.info(f"Figure saved: {output_path}")
    logger.info(f"Figure saved: {png_path}")
    return output_path, png_path


def _style_ax(ax, col: int):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("FPR", fontsize=10)
    if col == 0:
        ax.set_ylabel("TPR", fontsize=10)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=0.8)
    ax.grid(True, alpha=0.3)


def print_comparison_table(results_dir: str):
    """Print a formatted comparison table."""
    results_dir = Path(results_dir)
    divider = "=" * 74
    print(f"\n{divider}")
    print("  PIXEL_DIFF GENERALIZATION RESULTS  vs  Original NVFAIR Paper  (AUC)")
    print(divider)
    print(f"  {'Trained on':<16} {'Tested on':<16} {'PIXEL_DIFF':>10} "
          f"{'Paper':>8} {'Δ':>8}")
    print("-" * 74)

    for train_gen in GENERATORS:
        roc_file = results_dir / f"roc_data_{train_gen}.json"
        if not roc_file.exists():
            print(f"  {GEN_LABELS[train_gen]:<16} {'(no results)':}")
            continue
        with open(roc_file) as f:
            roc_data = json.load(f)

        for test_gen in GENERATORS:
            if test_gen not in roc_data:
                continue
            ours = roc_data[test_gen]["auc"]
            paper = PAPER_AUCS.get(train_gen, {}).get(test_gen)
            if paper is not None:
                delta = ours - paper
                delta_str = f"{delta:+.4f}"
            else:
                delta_str = "  N/A"
                paper = float("nan")
            print(
                f"  {GEN_LABELS[train_gen]:<16} {GEN_LABELS[test_gen]:<16} "
                f"{ours:>10.4f}  {paper:>8.4f}  {delta_str:>8}"
            )

    print(divider)


###############################################################################
#  MAIN
###############################################################################

def main():
    p = argparse.ArgumentParser(
        description="PIXEL_DIFF Generalization Experiment (NVFAIR Fig. 6 replica)"
    )

    p.add_argument("--nvfair_split_root", default="NVFAIR_split")
    p.add_argument("--frames_root", default="nvfair_frames")
    p.add_argument("--output_dir", default="outputs/gen_pixel_diff")
    p.add_argument("--results_dir", default="results/gen_pixel_diff")

    p.add_argument("--clip_length", type=int, default=64)
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--neighbor_k", type=int, default=4)

    p.add_argument("--num_epochs", type=int, default=50)
    p.add_argument("--steps_per_epoch", type=int, default=200)
    p.add_argument("--num_ids", type=int, default=16)
    p.add_argument("--vids_per_id", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--skip_training", action="store_true")
    p.add_argument("--skip_eval", action="store_true")
    p.add_argument("--plot_only", action="store_true")
    p.add_argument("--retrain", action="store_true")
    p.add_argument("--generators", nargs="+", default=GENERATORS, choices=GENERATORS)

    args = p.parse_args()

    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    checkpoints: Dict[str, str] = {}

    if not args.plot_only:
        if not args.skip_training:
            logger.info("=" * 60)
            logger.info("STEP 1 — Training one PIXEL_DIFF model per generator")
            logger.info("=" * 60)
            for gen in args.generators:
                ckpt = train_on_generator(args, gen)
                checkpoints[gen] = ckpt
        else:
            logger.info("STEP 1 — Skipped (--skip_training)")
            for gen in args.generators:
                ckpt = Path(args.output_dir) / f"gen_{gen}" / "best_model.pt"
                if ckpt.exists():
                    checkpoints[gen] = str(ckpt)
                    logger.info(f"  [{gen}] Using checkpoint: {ckpt}")
                else:
                    logger.warning(f"  [{gen}] Checkpoint NOT found: {ckpt}")

    if not args.plot_only and not args.skip_eval:
        logger.info("=" * 60)
        logger.info("STEP 2 — Cross-generator evaluation")
        logger.info("=" * 60)

        for gen, ckpt in checkpoints.items():
            results = evaluate_checkpoint_cross_generator(ckpt, args, gen)

            results_json = Path(args.results_dir) / f"gen_eval_{gen}.json"
            with open(results_json, "w") as f:
                json.dump(results, f, indent=2, default=str)

            roc_json = Path(args.results_dir) / f"roc_data_{gen}.json"
            with open(roc_json, "w") as f:
                json.dump(results["per_generator"], f, indent=2, default=str)

    logger.info("=" * 60)
    logger.info("STEP 3 — Plotting Fig. 6 style ROC figure")
    logger.info("=" * 60)

    out_pdf, out_png = plot_generalization_figure(
        results_dir=args.results_dir,
        output_path=str(Path(args.results_dir) / "fig6_generalization_pixel_diff.pdf"),
    )

    print_comparison_table(args.results_dir)

    logger.info("\nDone!")
    logger.info(f"  PDF → {out_pdf}")
    logger.info(f"  PNG → {out_png}")


if __name__ == "__main__":
    main()