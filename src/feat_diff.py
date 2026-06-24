#!/usr/bin/env python3
"""
ablation_clip_length.py  —  FULLY SELF-CONTAINED
=================================================
Clip-length ablation for FEAT_DIFF at T = 16, 32, 64, 128.
No imports from train_mol_avatar.py or ablation_dynamic_vs_appearance.py.
All model/dataset/loss/eval code is copied inline from
run_generalization_pixel_diff.py (which already works on your machine).

Usage:
  # 3 GPUs in parallel (T=16, T=32, T=128):
  bash run_clip_length_ablation.sh 0 1 2

  # Single run:
  CUDA_VISIBLE_DEVICES=0 python ablation_clip_length.py \
      --clip_length 32 --gpu 0 \
      --nvfair_split_root NVFAIR_split --frames_root nvfair_frames

  # Plot only (after runs finish):
  python ablation_clip_length.py --plot_only \
      --results_dir results/ablation_clip_length
"""

import argparse, json, logging, math, os, random, sys
from collections import defaultdict
from pathlib import Path
from typing import Dict

import cv2, numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm

try:
    from sklearn.metrics import roc_auc_score, roc_curve
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as e:
    sys.exit(f"Missing dependency: {e}")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

PAPER_T64  = {"overall": 0.877, "facevid2vid": 0.869, "tps": 0.880, "lia": 0.882}
COLORS     = {16: "#9C27B0", 32: "#E8624A", 64: "#2196F3", 128: "#5CB85C"}
LINESTYLES = {16: ":", 32: "--", 64: "-", 128: "-."}


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL — F5C backbone + FEAT_DIFF (copied from run_generalization_pixel_diff.py)
# ═══════════════════════════════════════════════════════════════════════════════

class ConvStack(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8,   4, stride=2), nn.BatchNorm2d(8),   nn.ReLU(inplace=True),
            nn.Conv2d(8, 32,  3, stride=2), nn.BatchNorm2d(32),  nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 2, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)


class FCC(nn.Module):
    def __init__(self, dim=64, meta_kernel_size=32, use_pe=True, bias=True):
        super().__init__()
        self.dim, self.use_pe = dim, use_pe
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
        return (self.mk_1_H[:,:,:s,:], self.mk_1_W[:,:,:,:s],
                self.mk_2_H[:,:,:s,:], self.mk_2_W[:,:,:,:s])

    def _pe(self, s):
        return (self.pe_1H[:,:,:s,:].expand(1,self.dim,s,s),
                self.pe_1W[:,:,:,:s].expand(1,self.dim,s,s),
                self.pe_2H[:,:,:s,:].expand(1,self.dim,s,s),
                self.pe_2W[:,:,:,:s].expand(1,self.dim,s,s))

    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=1)
        x1r, x2r = x1, x2
        _, _, s, _ = x1.shape
        K1H, K1W, K2H, K2W = self._k(s)
        if self.use_pe:
            p1H, p1W, p2H, p2W = self._pe(s)
            x1, x2 = x1 + p1H, x2 + p1W
        x1 = self.pre_norm_1(x1)
        x2 = self.pre_norm_2(x2)
        x1 = F.conv2d(torch.cat((x1, x1[:,:,:-1,:]), 2), weight=K1H, bias=self.b1H, groups=self.dim)
        x2 = F.conv2d(torch.cat((x2, x2[:,:,:,:-1]), 3), weight=K1W, bias=self.b1W, groups=self.dim)
        if self.use_pe:
            x1, x2 = x1 + p2W, x2 + p2H
        x1 = F.conv2d(torch.cat((x1, x1[:,:,:,:-1]), 3), weight=K2W, bias=self.b2W, groups=self.dim)
        x2 = F.conv2d(torch.cat((x2, x2[:,:,:-1,:]), 2), weight=K2H, bias=self.b2H, groups=self.dim)
        return torch.cat((x1r + x1, x2r + x2), dim=1)


def get_graph_feature(x, k=4):
    B, C, N = x.size()
    x_n = F.normalize(x, p=2, dim=1)
    sim = torch.bmm(x_n.transpose(1,2), x_n)
    idx = sim.topk(k=k, dim=-1)[1]
    base = torch.arange(B, device=x.device).view(-1,1,1) * N
    idx  = (idx + base).view(-1)
    xt   = x.transpose(1,2).contiguous()
    feat = xt.view(B*N,-1)[idx].view(B,N,k,C)
    xr   = xt.view(B,N,1,C).expand_as(feat)
    return (feat - xr).permute(0,3,1,2)


class CCC(nn.Module):
    def __init__(self):
        super().__init__()
        self.edge_fc = nn.Sequential(nn.Linear(128,128), nn.ReLU(inplace=True))
    def forward(self, f_i, f_j_k):
        return F.relu(f_i + f_j_k.max(dim=-1)[0], inplace=False)


class TemporalIdentityHead(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(128, 64, (3,3,3), stride=(1,2,1), padding=(1,1,1)),
            nn.BatchNorm3d(64), nn.ReLU(inplace=True),
            nn.Conv3d(64,  32, (3,3,3), stride=(1,2,1), padding=(1,1,1)),
            nn.BatchNorm3d(32), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((4,4,1)),
        )
        self.fc = nn.Sequential(
            nn.Linear(32*4*4, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.3), nn.Linear(256, embed_dim),
        )
    def forward(self, x):
        x = self.net(x).squeeze(-1).flatten(1)
        return F.normalize(self.fc(x), p=2, dim=-1)


class FeatDiffModel(nn.Module):
    """FEAT_DIFF: encode each frame with F5C, subtract consecutive features."""
    def __init__(self, embed_dim=256, neighbor_k=4):
        super().__init__()
        self.conv_stack = ConvStack()
        self.fcc = FCC(dim=64, meta_kernel_size=32)
        self.ccc = CCC()
        self.k   = neighbor_k
        self.head = TemporalIdentityHead(embed_dim=embed_dim)

    def _frame_feat(self, frame):
        f  = self.conv_stack(frame)
        f  = self.fcc(f)
        B, C, H, W = f.shape
        fl = f.reshape(B, C, H*W)
        gf = get_graph_feature(fl, k=self.k)
        ef = self.ccc(fl, gf)
        return (fl + ef).reshape(B, C, H, W)

    def forward(self, frames):
        """frames: (B, T, 1, H, W) → (B, embed_dim)"""
        B, T, C, H, W = frames.shape
        diffs, prev = [], None
        for t in range(T):
            cur = self._frame_feat(frames[:, t])
            if prev is not None:
                diffs.append(cur - prev)
            prev = cur
        vol = torch.stack(diffs, dim=-1)   # (B, 128, 16, 16, T-1)
        return self.head(vol)


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET  (copied verbatim from run_generalization_pixel_diff.py)
# ═══════════════════════════════════════════════════════════════════════════════

class NVFAIRFrameDataset(Dataset):
    def __init__(self, csv_path, frames_root, clip_length=64,
                 split='train', generator_filter=None):
        self.frames_root = Path(frames_root)
        self.clip_length = clip_length
        self.split       = split

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
            frame_dir = self.frames_root / row['new_path'].replace('.mp4', '')
            if not frame_dir.exists():
                skipped += 1; continue
            n_frames = len(list(frame_dir.glob('*.png')))
            if n_frames < 2:
                skipped += 1; continue

            driver  = str(row['driving_identity'])
            target  = str(row['target_identity'])
            is_self = bool(row['is_self_reenactment']) if 'is_self_reenactment' in row \
                      else (driver == target)

            idx = len(self.videos)
            self.videos.append({
                'frame_dir': str(frame_dir), 'num_frames': n_frames,
                'driving_identity': driver,
                'driving_identity_idx': self.driver_to_idx.get(driver, 0),
                'target_identity': target, 'is_self_reenactment': is_self,
                'generator': row.get('generator', 'unknown'),
            })
            self.identity_to_videos[driver]['self' if is_self else 'cross'].append(idx)

        if skipped:
            log.warning(f"Skipped {skipped} videos (no frames found)")
        log.info(f"NVFAIRFrameDataset [{split}]: {len(self.videos)} videos, "
                 f"{len(self.driver_to_idx)} identities")

    def __len__(self): return len(self.videos)

    def __getitem__(self, idx):
        v = self.videos[idx]
        files = sorted(Path(v['frame_dir']).glob('*.png'))
        n     = len(files)
        if self.split == 'train' and n > self.clip_length:
            start = random.randint(0, n - self.clip_length)
        elif n > self.clip_length:
            start = (n - self.clip_length) // 2
        else:
            start = 0
        sel  = files[start:start + self.clip_length]
        imgs = []
        for fp in sel:
            im = cv2.imread(str(fp), cv2.IMREAD_GRAYSCALE)
            if im is None: im = np.zeros((128,128), dtype=np.uint8)
            imgs.append(im.astype(np.float32) / 255.0)
        while len(imgs) < self.clip_length:
            imgs.append(imgs[-1])
        frames = torch.from_numpy(np.stack(imgs)[:, None, :, :])
        frames = (frames - frames.mean()) / (frames.std() + 1e-6)
        return {
            'frames': frames, 'length': min(n, self.clip_length),
            'driving_identity': v['driving_identity'],
            'driving_identity_idx': v['driving_identity_idx'],
            'target_identity': v['target_identity'],
            'is_self_reenactment': v['is_self_reenactment'],
            'generator': v['generator'],
        }


def collate_fn(batch):
    return {
        'frames':               torch.stack([b['frames'] for b in batch]),
        'lengths':              torch.tensor([b['length'] for b in batch]),
        'driving_identities':   [b['driving_identity'] for b in batch],
        'driving_identity_idxs': torch.tensor([b['driving_identity_idx'] for b in batch]),
        'target_identities':    [b['target_identity'] for b in batch],
        'is_self_reenactment':  torch.tensor([b['is_self_reenactment'] for b in batch]),
        'generators':           [b['generator'] for b in batch],
    }


class IdentityBatchSampler(Sampler):
    def __init__(self, dataset, num_identities=16, videos_per_identity=8, steps=200):
        self.ds       = dataset
        self.n_ids    = num_identities
        self.v_per_id = videos_per_identity
        self.steps    = steps
        self.valid_ids = [
            d for d, v in dataset.identity_to_videos.items()
            if len(v['self']) + len(v['cross']) >= videos_per_identity
        ]
        log.info(f"BatchSampler: {len(self.valid_ids)} valid identities")

    def __iter__(self):
        for _ in range(self.steps):
            ids   = random.sample(self.valid_ids, min(self.n_ids, len(self.valid_ids)))
            batch = []
            for d in ids:
                pool = (self.ds.identity_to_videos[d]['self'] +
                        self.ds.identity_to_videos[d]['cross'])
                batch.extend(random.sample(pool, self.v_per_id)
                             if len(pool) >= self.v_per_id
                             else random.choices(pool, k=self.v_per_id))
            yield batch

    def __len__(self): return self.steps


# ═══════════════════════════════════════════════════════════════════════════════
# LOSS & EVALUATION  (copied from run_generalization_pixel_diff.py)
# ═══════════════════════════════════════════════════════════════════════════════

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__(); self.t = temperature
    def forward(self, emb, labels):
        B, device = emb.shape[0], emb.device
        emb = F.normalize(emb, p=2, dim=1)
        sim = emb @ emb.T / self.t
        eye = torch.eye(B, device=device)
        pos = (labels.view(-1,1) == labels.view(1,-1)).float() - eye
        sim = sim - sim.max(dim=1, keepdim=True)[0].detach()
        exp = torch.exp(sim) * (1 - eye)
        log_p = sim - torch.log(exp.sum(1, keepdim=True) + 1e-8)
        n_pos = pos.sum(1).clamp(min=1)
        return -(pos * log_p).sum(1).div(n_pos).mean()


def compute_auc_paper_method(embeddings, target_ids, is_self):
    embeddings = F.normalize(embeddings, p=2, dim=1)
    per_target = {}
    for target in sorted(set(target_ids)):
        mask = torch.tensor([t == target for t in target_ids])
        idx  = torch.where(mask)[0]
        if len(idx) < 2: continue
        e, sf = embeddings[idx], is_self[idx]
        si, ci = torch.where(sf)[0], torch.where(~sf)[0]
        if len(si) < 2 or len(ci) < 1: continue
        scores, labels = [], []
        for i in range(len(si)):
            for j in range(i+1, len(si)):
                scores.append(-torch.norm(e[si[i]]-e[si[j]], p=2).item()); labels.append(1)
        for s in si:
            for c in ci:
                scores.append(-torch.norm(e[s]-e[c], p=2).item()); labels.append(0)
        if len(set(labels)) < 2: continue
        try: per_target[target] = roc_auc_score(labels, scores)
        except: pass
    return (float(np.mean(list(per_target.values()))) if per_target else 0.5), per_target


def extract_all_embeddings(model, loader, device):
    model.eval()
    embs, tgts, sfs, gens = [], [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Embed", leave=False):
            e = model(batch['frames'].to(device))
            embs.append(e.cpu())
            tgts.extend(batch['target_identities'])
            sfs.append(batch['is_self_reenactment'])
            gens.extend(batch['generators'])
    return {
        'embeddings': torch.cat(embs),
        'target_ids': tgts,
        'is_self':    torch.cat(sfs).bool(),
        'generators': gens,
    }


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def train(args) -> str:
    T, label = args.clip_length, f"T{args.clip_length}"
    ckpt_dir  = Path(args.output_dir) / label
    ckpt_path = ckpt_dir / "best_model.pt"

    if ckpt_path.exists() and not args.retrain:
        log.info(f"[{label}] Checkpoint exists — skipping"); return str(ckpt_path)

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    log.info(f"[{label}] device={device}")

    root = Path(args.nvfair_split_root)
    train_ds = NVFAIRFrameDataset(str(root/"train_files.csv"), args.frames_root,
                                   clip_length=T, split='train')
    val_ds   = NVFAIRFrameDataset(str(root/"val_files.csv"),   args.frames_root,
                                   clip_length=T, split='val')
    if len(train_ds) == 0:
        sys.exit(f"[{label}] No training data. Check paths.")

    sampler = IdentityBatchSampler(train_ds, num_identities=args.num_ids,
                                   videos_per_identity=args.vids_per_id,
                                   steps=args.steps_per_epoch)
    train_loader = DataLoader(train_ds, batch_sampler=sampler,
                              num_workers=args.num_workers, collate_fn=collate_fn,
                              pin_memory=True, persistent_workers=(args.num_workers>0))
    val_loader   = DataLoader(val_ds, batch_size=16, shuffle=False,
                              num_workers=4, collate_fn=collate_fn, pin_memory=True)

    model   = FeatDiffModel(embed_dim=args.embed_dim, neighbor_k=args.neighbor_k).to(device)
    n       = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"[{label}] FeatDiffModel  {n:,} params  T={T}")

    loss_fn = SupConLoss(temperature=args.temperature)
    optim   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    total_steps, warmup_steps = args.num_epochs*args.steps_per_epoch, 5*args.steps_per_epoch

    def lr_fn(step):
        if step < warmup_steps: return step / max(1, warmup_steps)
        prog = (step-warmup_steps) / max(1, total_steps-warmup_steps)
        return max(0.01, 0.5*(1.0+math.cos(math.pi*prog)))

    sched  = torch.optim.lr_scheduler.LambdaLR(optim, lr_fn)
    scaler = GradScaler()
    best_auc, history = 0.0, []

    config = dict(clip_length=T, embed_dim=args.embed_dim, neighbor_k=args.neighbor_k,
                  lr=args.lr, wd=args.wd, temperature=args.temperature,
                  num_epochs=args.num_epochs, steps_per_epoch=args.steps_per_epoch)
    with open(ckpt_dir/"config.json","w") as f: json.dump(config, f, indent=2)

    log.info(f"[{label}] {args.num_epochs} epochs × {args.steps_per_epoch} steps  "
             f"batch={args.num_ids}×{args.vids_per_id}={args.num_ids*args.vids_per_id}")

    for epoch in range(args.num_epochs):
        model.train(); total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"[{label}] Ep {epoch+1}/{args.num_epochs}", leave=False):
            frames = batch['frames'].to(device, non_blocking=True)
            labels = batch['driving_identity_idxs'].to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            with autocast():
                emb  = model(frames)
                loss = loss_fn(emb, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim); scaler.update(); sched.step()
            total_loss += loss.item()

        avg_loss = total_loss / args.steps_per_epoch
        if (epoch+1) % 5 == 0 or epoch == 0:
            vd  = extract_all_embeddings(model, val_loader, device)
            auc, _ = compute_auc_paper_method(vd['embeddings'], vd['target_ids'], vd['is_self'])
            log.info(f"[{label}] Epoch {epoch+1:3d}  loss={avg_loss:.4f}  val_AUC={auc:.4f}")
            history.append({"epoch": epoch+1, "loss": avg_loss, "val_auc": auc})
            if auc > best_auc:
                best_auc = auc
                torch.save({"model": model.state_dict(), "auc": auc,
                            "epoch": epoch, "config": config}, ckpt_path)
                log.info(f"[{label}]  ★ New best val_AUC={auc:.4f}")
        else:
            log.info(f"[{label}] Epoch {epoch+1:3d}  loss={avg_loss:.4f}")
            history.append({"epoch": epoch+1, "loss": avg_loss, "val_auc": None})

    with open(ckpt_dir/"training_history.json","w") as f: json.dump(history, f, indent=2)
    log.info(f"[{label}] Done. Best val_AUC={best_auc:.4f}")
    return str(ckpt_path)


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate(ckpt_path: str, args, T: int) -> dict:
    label  = f"T{T}"
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    ckpt   = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg    = ckpt.get("config", {})

    model = FeatDiffModel(embed_dim=cfg.get("embed_dim", args.embed_dim),
                          neighbor_k=cfg.get("neighbor_k", args.neighbor_k)).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()

    root    = Path(args.nvfair_split_root)
    test_ds = NVFAIRFrameDataset(str(root/"test_files.csv"), args.frames_root,
                                  clip_length=cfg.get("clip_length", T), split='test')
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False,
                             num_workers=4, collate_fn=collate_fn, pin_memory=True)

    log.info(f"[eval/{label}] {len(test_ds)} test videos …")
    td = extract_all_embeddings(model, test_loader, device)

    overall_auc, per_target = compute_auc_paper_method(
        td['embeddings'], td['target_ids'], td['is_self'])

    per_gen = {}
    for gen in ["facevid2vid","tps","lia"]:
        mask = torch.tensor([g == gen for g in td['generators']])
        idx  = torch.where(mask)[0]
        if len(idx) < 10: continue
        g_auc, _ = compute_auc_paper_method(
            td['embeddings'][idx], [td['target_ids'][i] for i in idx.tolist()],
            td['is_self'][idx])
        per_gen[gen] = float(g_auc)
        log.info(f"[eval/{label}] {gen}: {g_auc:.4f}")

    # ROC curve
    emb = F.normalize(td['embeddings'], p=2, dim=1)
    all_scores, all_labels = [], []
    for target in sorted(set(td['target_ids'])):
        mask = torch.tensor([t == target for t in td['target_ids']])
        idx  = torch.where(mask)[0]
        e, sf = emb[idx], td['is_self'][idx]
        si, ci = torch.where(sf)[0], torch.where(~sf)[0]
        if len(si) < 2 or len(ci) < 1: continue
        for i in range(len(si)):
            for j in range(i+1, len(si)):
                all_scores.append(-torch.norm(e[si[i]]-e[si[j]],p=2).item()); all_labels.append(1)
        for s in si:
            for c in ci:
                all_scores.append(-torch.norm(e[s]-e[c],p=2).item()); all_labels.append(0)

    fpr, tpr, _ = roc_curve(all_labels, all_scores)
    log.info(f"[eval/{label}] Overall AUC={overall_auc:.4f}")
    return {"clip_length": T, "overall_auc": float(overall_auc),
            "per_generator": per_gen, "n_targets": len(per_target),
            "val_auc": float(ckpt.get("auc", 0.0)),
            "fpr": fpr.tolist(), "tpr": tpr.tolist()}


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT + LATEX TABLE
# ═══════════════════════════════════════════════════════════════════════════════

def plot(results: Dict[int, dict], results_dir: str):
    results_dir = Path(results_dir); results_dir.mkdir(parents=True, exist_ok=True)
    fig, axes   = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    for T in sorted(results.keys()):
        r = results[T]; auc = r["overall_auc"]
        ax.plot(np.array(r["fpr"]), np.array(r["tpr"]),
                color=COLORS[T], lw=2.5 if T==64 else 1.8,
                linestyle=LINESTYLES[T], label=f"T={T}  (AUC={auc:.3f})",
                zorder=3 if T==64 else 2)
    ax.plot([0,1],[0,1],"k--",alpha=0.3,lw=0.8)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("ROC — Clip-Length Ablation (FEAT DIFF)")
    ax.legend(fontsize=9, loc="lower right"); ax.grid(alpha=0.3)
    ax.set_xlim(0,1); ax.set_ylim(0,1)

    ax2   = axes[1]
    gens  = ["facevid2vid","tps","lia"]
    gl    = {"facevid2vid":"FV2V","tps":"TPS","lia":"LIA"}
    Ts    = sorted(results.keys()); x = np.arange(len(gens))
    w     = 0.18; offs = np.linspace(-(len(Ts)-1)/2,(len(Ts)-1)/2,len(Ts))
    for i, T in enumerate(Ts):
        vals = [results[T]["per_generator"].get(g,0.0) for g in gens]
        bars = ax2.bar(x+offs[i]*w, vals, w, label=f"T={T}",
                       color=COLORS[T], alpha=0.85, edgecolor="white")
        for bar, v in zip(bars, vals):
            if v > 0:
                ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
                         f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    ax2.set_xticks(x); ax2.set_xticklabels([gl[g] for g in gens])
    ax2.set_ylabel("AUC"); ax2.set_title("Per-Generator AUC vs Clip Length")
    ax2.set_ylim(0.5,1.0); ax2.legend(fontsize=9); ax2.grid(axis="y",alpha=0.3)

    plt.tight_layout()
    for ext in ("pdf","png"):
        p = results_dir/f"fig_clip_length_ablation.{ext}"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        log.info(f"Saved → {p}")
    plt.close()


def latex_table(results: Dict[int, dict]):
    print("\n% ── Table: Clip-Length Ablation ───────────────────────────")
    print("\\begin{table}[t]")
    print("\\caption{Clip-length ablation (FEAT~DIFF, F5C backbone, 30-epoch")
    print("schedule). $T{=}64$ is the paper default (Table~\\ref{tab:main}).}")
    print("\\label{tab:clip_length}")
    print("\\centering\\small")
    print("\\begin{tabular}{ccccc}\\toprule")
    print("$T$ & Overall & FV2V & TPS & LIA \\\\\\midrule")
    for T in sorted(results.keys()):
        r = results[T]
        ov = r["overall_auc"]; fv = r["per_generator"].get("facevid2vid",0.0)
        tp = r["per_generator"].get("tps",0.0); li = r["per_generator"].get("lia",0.0)
        f  = lambda v: f"\\textbf{{{v:.3f}}}" if T==64 else f"{v:.3f}"
        print(f"${T}$ & {f(ov)} & {f(fv)} & {f(tp)} & {f(li)} \\\\")
    print("\\bottomrule\\end{tabular}\\end{table}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--nvfair_split_root", default="NVFAIR_split")
    p.add_argument("--frames_root",       default="nvfair_frames")
    p.add_argument("--output_dir",        default="outputs/ablation_clip_length")
    p.add_argument("--results_dir",       default="results/ablation_clip_length")
    p.add_argument("--clip_length", type=int, default=None, choices=[16,32,64,128])
    p.add_argument("--gpu",         type=int, default=0)
    p.add_argument("--num_epochs",      type=int,   default=30)
    p.add_argument("--steps_per_epoch", type=int,   default=200)
    p.add_argument("--num_ids",         type=int,   default=16)
    p.add_argument("--vids_per_id",     type=int,   default=8)
    p.add_argument("--lr",              type=float, default=1e-3)
    p.add_argument("--wd",              type=float, default=1e-4)
    p.add_argument("--temperature",     type=float, default=0.07)
    p.add_argument("--embed_dim",       type=int,   default=256)
    p.add_argument("--neighbor_k",      type=int,   default=4)
    p.add_argument("--num_workers",     type=int,   default=4)
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--retrain",   action="store_true")
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--plot_only", action="store_true")
    return p.parse_args()


def main():
    args        = get_args()
    results_dir = Path(args.results_dir); results_dir.mkdir(parents=True, exist_ok=True)
    json_path   = results_dir / "clip_length_ablation_results.json"

    if args.plot_only:
        if not json_path.exists(): sys.exit(f"Not found: {json_path}")
        with open(json_path) as f: results = {int(k):v for k,v in json.load(f).items()}
        plot(results, args.results_dir); latex_table(results); return

    Ts = [args.clip_length] if args.clip_length else [16, 32, 64, 128]
    results = {}
    if json_path.exists():
        with open(json_path) as f: results = {int(k):v for k,v in json.load(f).items()}

    for T in Ts:
        args.clip_length = T
        log.info(f"\n{'='*55}\nCLIP LENGTH T={T}\n{'='*55}")
        if not args.eval_only:
            ckpt = train(args)
        else:
            ckpt = str(Path(args.output_dir)/f"T{T}"/"best_model.pt")
            if not Path(ckpt).exists(): log.error(f"Missing: {ckpt}"); continue
        results[T] = evaluate(ckpt, args, T)
        with open(json_path,"w") as f:
            json.dump({str(k):v for k,v in results.items()}, f, indent=2)

    log.info("\n" + "="*55)
    log.info(f"{'T':>5}  {'Overall':>8}  {'FV2V':>7}  {'TPS':>7}  {'LIA':>7}")
    for T in sorted(results.keys()):
        r = results[T]
        log.info(f"  T={T:<4}  {r['overall_auc']:.4f}    "
                 f"{r['per_generator'].get('facevid2vid',0):.4f}   "
                 f"{r['per_generator'].get('tps',0):.4f}   "
                 f"{r['per_generator'].get('lia',0):.4f}")
    log.info(f"  T=64   {PAPER_T64['overall']:.4f}    "
             f"{PAPER_T64['facevid2vid']:.4f}   {PAPER_T64['tps']:.4f}   "
             f"{PAPER_T64['lia']:.4f}  (Table III, 150 epochs)")

    if len(results) >= 2:
        plot(results, args.results_dir); latex_table(results)
    log.info(f"\nResults → {json_path}")


if __name__ == "__main__":
    main()