#!/usr/bin/env python3
"""
train_resnet18_all_variants.py
==============================
All 4 ResNet18 conditions matching the F5C ablation (Table VI):
  pixel_diff  — pixel-space diff → ResNet18  (working, no collapse)
  raw_feat    — raw frame features, mean pool (appearance + motion)
  static      — single center frame only (appearance baseline)
  feat_diff   — feature-space diff WITH L2-norm fix (avoids collapse)

WHY FEAT_DIFF COLLAPSED BEFORE AND HOW IT IS FIXED
---------------------------------------------------
Problem: ResNet18 encodes appearance. Consecutive talking-head frames
look almost identical → feat_{t+1} ≈ feat_t → diff ≈ 0 → head receives
zeros → all embeddings collapse → AUC ≈ 0.5.

Fix: L2-normalize each frame feature BEFORE differencing.
  feat_t_norm = feat_t / ||feat_t||   (on unit hypersphere)
  diff_t = feat_{t+1}_norm - feat_t_norm

On the unit sphere, even tiny angular differences produce non-zero diffs.
The backbone is forced to learn features where consecutive frames ARE
different (otherwise the diff carries no identity signal and loss stays high).
This creates a gradient signal that pushes the backbone to encode
motion-sensitive features rather than pure appearance.

This is the same principle as SimSiam / BYOL stop-gradient tricks —
preventing trivial solutions by operating in normalized space.

Usage (one command per GPU):
  CUDA_VISIBLE_DEVICES=4 python train_resnet18_all_variants.py --variant feat_diff  --gpu 0 &
  CUDA_VISIBLE_DEVICES=5 python train_resnet18_all_variants.py --variant raw_feat   --gpu 0 &
  CUDA_VISIBLE_DEVICES=6 python train_resnet18_all_variants.py --variant static     --gpu 0 &
"""

import argparse, json, logging, math, random, sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2, numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm

try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    sys.exit("pip install scikit-learn")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)
cv2.setNumThreads(0)


# ═══════════════════════════════════════════════════════════════════════════════
# BACKBONE
# ═══════════════════════════════════════════════════════════════════════════════

def _build_resnet18():
    """ResNet18 from scratch, grayscale 1-channel input."""
    from torchvision.models import resnet18
    base = resnet18(weights=None)
    c1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
    nn.init.kaiming_normal_(c1.weight, mode='fan_out', nonlinearity='relu')
    base.conv1 = c1
    return nn.Sequential(
        base.conv1, base.bn1, base.relu, base.maxpool,
        base.layer1, base.layer2, base.layer3, base.layer4, base.avgpool,
    )  # output: (B, 512, 1, 1)


def _head(embed_dim=256):
    return nn.Sequential(
        nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
        nn.Dropout(0.3), nn.Linear(512, embed_dim),
    )


def _encode_chunked(backbone, x, chunk_size):
    """Encode (N, C, H, W) through backbone in chunks. Returns (N, 512)."""
    out = []
    for i in range(0, x.shape[0], chunk_size):
        out.append(backbone(x[i:i+chunk_size]).flatten(1))
    return torch.cat(out)


# ═══════════════════════════════════════════════════════════════════════════════
# FOUR MODEL VARIANTS
# ═══════════════════════════════════════════════════════════════════════════════

class ResNet18FeatDiff(nn.Module):
    """
    Feature-space differencing WITH L2-normalization fix.

    feat_t      = ResNet18(frame_t)           shape: (B, 512)
    feat_t_norm = L2_normalize(feat_t)        → unit sphere
    diff_t      = feat_{t+1}_norm - feat_t_norm   → non-zero even for similar frames
    pool        = mean(diff_0 ... diff_{T-2})
    embed       = MLP(pool) → L2-norm → 256-d

    Key insight: operating on the unit sphere means the backbone MUST learn
    diverse frame features (otherwise diffs = 0 and loss cannot decrease).
    """
    def __init__(self, embed_dim=256):
        super().__init__()
        self.backbone = _build_resnet18()
        self.head     = _head(embed_dim)
        self.embed_dim = embed_dim
        log.info(f"ResNet18FeatDiff (L2-norm fix): "
                 f"{sum(p.numel() for p in self.parameters())/1e6:.2f}M params")

    def forward(self, frames, chunk_size=128):
        B, T, C, H, W = frames.shape
        flat  = frames.reshape(B*T, C, H, W)
        feats = _encode_chunked(self.backbone, flat, chunk_size)  # (B*T, 512)
        # *** THE FIX: L2-normalize before differencing ***
        feats = F.normalize(feats.view(B, T, 512), p=2, dim=-1)   # (B, T, 512) unit sphere
        diffs = feats[:, 1:] - feats[:, :-1]                      # (B, T-1, 512)
        pooled = diffs.mean(dim=1)                                 # (B, 512)
        return F.normalize(self.head(pooled), p=2, dim=-1)


class ResNet18PixelDiff(nn.Module):
    """Pixel-space diff → ResNet18. Already working."""
    def __init__(self, embed_dim=256):
        super().__init__()
        self.backbone = _build_resnet18()
        self.head     = _head(embed_dim)
        self.embed_dim = embed_dim
        log.info(f"ResNet18PixelDiff: "
                 f"{sum(p.numel() for p in self.parameters())/1e6:.2f}M params")

    def forward(self, frames, chunk_size=128):
        B, T, C, H, W = frames.shape
        diffs = (frames[:, 1:] - frames[:, :-1]).reshape(B*(T-1), C, H, W)
        feats = _encode_chunked(self.backbone, diffs, chunk_size).view(B, T-1, 512)
        return F.normalize(self.head(feats.mean(dim=1)), p=2, dim=-1)


class ResNet18RawFeat(nn.Module):
    """Raw frame features, mean pooled. Appearance + motion, no diff."""
    def __init__(self, embed_dim=256):
        super().__init__()
        self.backbone = _build_resnet18()
        self.head     = _head(embed_dim)
        self.embed_dim = embed_dim
        log.info(f"ResNet18RawFeat: "
                 f"{sum(p.numel() for p in self.parameters())/1e6:.2f}M params")

    def forward(self, frames, chunk_size=128):
        B, T, C, H, W = frames.shape
        feats = _encode_chunked(self.backbone,
                                frames.reshape(B*T, C, H, W),
                                chunk_size).view(B, T, 512)
        return F.normalize(self.head(feats.mean(dim=1)), p=2, dim=-1)


class ResNet18Static(nn.Module):
    """Single center frame only. Appearance baseline — should be near 0.5."""
    def __init__(self, embed_dim=256):
        super().__init__()
        self.backbone = _build_resnet18()
        self.head     = _head(embed_dim)
        self.embed_dim = embed_dim
        log.info(f"ResNet18Static: "
                 f"{sum(p.numel() for p in self.parameters())/1e6:.2f}M params")

    def forward(self, frames, chunk_size=128):
        B, T, C, H, W = frames.shape
        center = frames[:, T//2]                                   # (B, 1, H, W)
        feat   = self.backbone(center).flatten(1)                  # (B, 512)
        return F.normalize(self.head(feat), p=2, dim=-1)


MODELS = {
    'feat_diff':  ResNet18FeatDiff,
    'pixel_diff': ResNet18PixelDiff,
    'raw_feat':   ResNet18RawFeat,
    'static':     ResNet18Static,
}


# ═══════════════════════════════════════════════════════════════════════════════
# LOSS
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


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET  (identical to ablation_clip_length.py)
# ═══════════════════════════════════════════════════════════════════════════════

class NVFAIRFrameDataset(Dataset):
    def __init__(self, csv_path, frames_root, clip_length=64, split='train'):
        self.frames_root = Path(frames_root)
        self.clip_length = clip_length
        self.split       = split
        df = pd.read_csv(csv_path,
                         dtype={'target_identity': str, 'driving_identity': str},
                         low_memory=False)
        drivers = sorted(df['driving_identity'].unique())
        self.d2i = {str(d): i for i, d in enumerate(drivers)}
        self.videos = []
        self.identity_to_videos = defaultdict(lambda: {'self': [], 'cross': []})
        skipped = 0
        for _, row in df.iterrows():
            fd = self.frames_root / str(row['new_path']).replace('.mp4', '')
            if not fd.exists(): skipped += 1; continue
            n = len(list(fd.glob('*.png')))
            if n < 2: skipped += 1; continue
            driver  = str(row['driving_identity'])
            target  = str(row['target_identity'])
            is_self = bool(row['is_self_reenactment']) \
                      if 'is_self_reenactment' in row else (driver == target)
            idx = len(self.videos)
            self.videos.append({'frame_dir': str(fd), 'n': n,
                                'driver': driver, 'driver_idx': self.d2i[driver],
                                'target': target, 'is_self': is_self,
                                'generator': str(row.get('generator', ''))})
            self.identity_to_videos[driver]['self' if is_self else 'cross'].append(idx)
        self.valid_drivers = [d for d, v in self.identity_to_videos.items()
                              if v['self'] and v['cross']]
        if skipped: log.warning(f"Skipped {skipped} (no frames)")
        log.info(f"[{split}] {len(self.videos)} videos, "
                 f"{len(self.d2i)} drivers, {len(self.valid_drivers)} valid")

    def __len__(self): return len(self.videos)

    def __getitem__(self, idx):
        v = self.videos[idx]
        files = sorted(Path(v['frame_dir']).glob('*.png'))
        n = len(files)
        start = (random.randint(0, n-self.clip_length)
                 if self.split == 'train' and n > self.clip_length
                 else (n-self.clip_length)//2 if n > self.clip_length else 0)
        sel = files[start: start+self.clip_length]
        imgs = []
        for fp in sel:
            im = cv2.imread(str(fp), cv2.IMREAD_GRAYSCALE)
            if im is None: im = np.zeros((128, 128), dtype=np.uint8)
            imgs.append(im.astype(np.float32)/255.0)
        while len(imgs) < self.clip_length: imgs.append(imgs[-1])
        frames = torch.from_numpy(np.stack(imgs)[:, None, :, :])
        frames = (frames - frames.mean()) / (frames.std() + 1e-6)
        return {'frames': frames, 'driver_idx': v['driver_idx'],
                'target': v['target'], 'is_self': v['is_self'],
                'generator': v['generator']}


def collate_fn(batch):
    return {
        'frames':      torch.stack([b['frames']      for b in batch]),
        'driver_idxs': torch.tensor([b['driver_idx'] for b in batch]),
        'target_ids':  [b['target']    for b in batch],
        'is_self':     torch.tensor([b['is_self']    for b in batch]),
        'generators':  [b['generator'] for b in batch],
    }


class IdentityBatchSampler(Sampler):
    def __init__(self, ds, num_ids=8, vids_per_id=4, steps=200):
        self.ds, self.n, self.m, self.steps = ds, num_ids, vids_per_id, steps
        self.valid = ds.valid_drivers or list(ds.identity_to_videos.keys())
        log.info(f"Sampler: {len(self.valid)} drivers, "
                 f"batch={num_ids}×{vids_per_id}={num_ids*vids_per_id}")

    def __len__(self): return self.steps

    def __iter__(self):
        for _ in range(self.steps):
            ids = (random.sample(self.valid, min(self.n, len(self.valid)))
                   if len(self.valid) >= self.n
                   else random.choices(self.valid, k=self.n))
            batch = []
            for d in ids:
                pool = (self.ds.identity_to_videos[d]['self'] +
                        self.ds.identity_to_videos[d]['cross'])
                batch += (random.sample(pool, self.m) if len(pool) >= self.m
                          else random.choices(pool, k=self.m))
            yield batch


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

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
        sc, lb = [], []
        for i in range(len(si)):
            for j in range(i+1, len(si)):
                sc.append(-torch.norm(e[si[i]]-e[si[j]], p=2).item()); lb.append(1)
        for s in si:
            for c in ci:
                sc.append(-torch.norm(e[s]-e[c], p=2).item()); lb.append(0)
        if len(set(lb)) < 2: continue
        try: per_target[target] = roc_auc_score(lb, sc)
        except: pass
    return (float(np.mean(list(per_target.values()))) if per_target else 0.5), per_target


@torch.no_grad()
def extract_embeddings(model, loader, device, chunk_size):
    model.eval()
    embs, tgts, sfs, gens = [], [], [], []
    for batch in tqdm(loader, desc="Embed", leave=False):
        with autocast('cuda'):
            e = model(batch['frames'].to(device), chunk_size=chunk_size)
        embs.append(e.cpu()); tgts.extend(batch['target_ids'])
        sfs.append(batch['is_self']); gens.extend(batch['generators'])
    return {'embeddings': torch.cat(embs), 'target_ids': tgts,
            'is_self': torch.cat(sfs).bool(), 'generators': gens}


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def train(args):
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    ts  = datetime.now().strftime('%Y%m%d_%H%M%S')
    run = f"rn18_{args.variant}_{ts}"
    out = Path(args.output_dir) / run
    out.mkdir(parents=True, exist_ok=True)
    with open(out/'args.json','w') as f: json.dump(vars(args), f, indent=2)

    root = Path(args.nvfair_split_root)
    log.info(f"Loading datasets [variant={args.variant}]...")
    train_ds = NVFAIRFrameDataset(str(root/'train_files.csv'), args.frames_root,
                                   args.clip_length, split='train')
    val_ds   = NVFAIRFrameDataset(str(root/'val_files.csv'),   args.frames_root,
                                   args.clip_length, split='val')
    test_ds  = NVFAIRFrameDataset(str(root/'test_files.csv'),  args.frames_root,
                                   args.clip_length, split='test')

    sampler = IdentityBatchSampler(train_ds, args.num_ids, args.vids_per_id,
                                   args.steps_per_epoch)
    train_loader = DataLoader(train_ds, batch_sampler=sampler,
                              num_workers=args.num_workers, collate_fn=collate_fn,
                              pin_memory=True, persistent_workers=(args.num_workers>0))
    val_loader  = DataLoader(val_ds,  batch_size=16, shuffle=False,
                             num_workers=4, collate_fn=collate_fn, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False,
                             num_workers=4, collate_fn=collate_fn, pin_memory=True)

    model   = MODELS[args.variant](embed_dim=args.embed_dim).to(device)
    loss_fn = SupConLoss(temperature=args.temperature)
    optim   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scaler  = GradScaler('cuda')

    total_steps  = args.num_epochs * args.steps_per_epoch
    warmup_steps = 5 * args.steps_per_epoch

    def lr_fn(step):
        if step < warmup_steps: return step / max(1, warmup_steps)
        prog = (step-warmup_steps) / max(1, total_steps-warmup_steps)
        return max(0.01, 0.5*(1+math.cos(math.pi*prog)))

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_fn)

    log.info("="*65)
    log.info(f"Run: {run}")
    log.info(f"  variant={args.variant}  T={args.clip_length}  "
             f"loss=SupCon(τ={args.temperature})")
    log.info(f"  {args.num_epochs} epochs × {args.steps_per_epoch} steps  "
             f"batch={args.num_ids}×{args.vids_per_id}={args.num_ids*args.vids_per_id}")
    log.info("="*65)

    best_val_auc = 0.0; history = []; step = 0

    for epoch in range(1, args.num_epochs+1):
        model.train(); total_loss = 0.0
        for batch in tqdm(train_loader,
                          desc=f"Ep {epoch}/{args.num_epochs}", leave=False):
            frames = batch['frames'].to(device, non_blocking=True)
            labels = batch['driver_idxs'].to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            with autocast('cuda'):
                emb  = model(frames, chunk_size=args.chunk_size)
                loss = loss_fn(emb, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim); scaler.update()
            step += 1; sched.step()
            total_loss += loss.item()

        avg_loss = total_loss / args.steps_per_epoch

        if epoch % 5 == 0 or epoch == 1 or epoch == args.num_epochs:
            vd  = extract_embeddings(model, val_loader, device, args.chunk_size)
            auc, _ = compute_auc_paper_method(vd['embeddings'],
                                              vd['target_ids'], vd['is_self'])
            log.info(f"Epoch {epoch:3d}  loss={avg_loss:.4f}  val_AUC={auc:.4f}")
            history.append({'epoch': epoch, 'loss': avg_loss, 'val_auc': auc})
            if auc > best_val_auc:
                best_val_auc = auc
                torch.save({'model': model.state_dict(), 'val_auc': auc,
                            'epoch': epoch, 'variant': args.variant},
                           out/'best_model.pt')
                log.info(f"  ★ New best val_AUC={auc:.4f}")
        else:
            log.info(f"Epoch {epoch:3d}  loss={avg_loss:.4f}")
            history.append({'epoch': epoch, 'loss': avg_loss, 'val_auc': None})

    with open(out/'history.json','w') as f: json.dump(history, f, indent=2)

    # ── Final test ────────────────────────────────────────────────────────────
    log.info(f"\n── Final Test [{args.variant}] ──")
    ckpt = torch.load(out/'best_model.pt', map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model']); model.eval()
    td = extract_embeddings(model, test_loader, device, args.chunk_size)
    overall, _ = compute_auc_paper_method(td['embeddings'],
                                          td['target_ids'], td['is_self'])
    per_gen = {}
    for g in ['facevid2vid','tps','lia']:
        mask = torch.tensor([x == g for x in td['generators']])
        idx  = torch.where(mask)[0]
        if len(idx) < 10: continue
        ga, _ = compute_auc_paper_method(
            td['embeddings'][idx], [td['target_ids'][i] for i in idx.tolist()],
            td['is_self'][idx])
        per_gen[g] = float(ga)

    log.info(f"\n{'='*55}")
    log.info(f"ResNet18+{args.variant} | Overall={overall:.4f} | "
             f"FV2V={per_gen.get('facevid2vid',0):.4f} | "
             f"TPS={per_gen.get('tps',0):.4f} | "
             f"LIA={per_gen.get('lia',0):.4f}")
    log.info(f"F5C+FeatDiff (Table III):    0.877       0.869       0.880       0.882")
    log.info(f"{'='*55}")

    with open(out/'test_results.json','w') as f:
        json.dump({'overall': float(overall), 'per_gen': per_gen,
                   'best_val_auc': best_val_auc, 'variant': args.variant}, f, indent=2)
    log.info(f"Done → {out}/test_results.json")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--nvfair_split_root', default='NVFAIR_split')
    p.add_argument('--frames_root',        default='nvfair_frames')
    p.add_argument('--output_dir',         default='outputs_resnet18')
    p.add_argument('--variant', default='feat_diff',
                   choices=['feat_diff','pixel_diff','raw_feat','static'])
    p.add_argument('--clip_length',    type=int,   default=64)
    p.add_argument('--embed_dim',      type=int,   default=256)
    p.add_argument('--temperature',    type=float, default=0.07)
    p.add_argument('--num_epochs',     type=int,   default=150)
    p.add_argument('--steps_per_epoch',type=int,   default=200)
    p.add_argument('--num_ids',        type=int,   default=8)
    p.add_argument('--vids_per_id',    type=int,   default=4)
    p.add_argument('--chunk_size',     type=int,   default=128)
    p.add_argument('--lr',             type=float, default=1e-3)
    p.add_argument('--wd',             type=float, default=1e-4)
    p.add_argument('--num_workers',    type=int,   default=4)
    p.add_argument('--gpu',            type=int,   default=0)
    p.add_argument('--seed',           type=int,   default=42)
    args = p.parse_args()
    train(args)