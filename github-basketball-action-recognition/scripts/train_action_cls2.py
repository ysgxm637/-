# -*- coding: utf-8 -*-
"""二阶段 2.0 分类器赛马训练: R(2+1)D-18 multi-seed 基线 / MViT V2-S / R(2+1)D+BiGRU 头。

数据: dataset_action_clip_val35_sliding (train 12375 / val 6824, 16帧 jpg clip)
与 1.0 train_action_clip.py 完全同源: sqrt 采样 / 时序一致增广 / AdamW+cosine /
label_smoothing 0.05 / val macro-F1 选优早停(PATIENCE=8, EPOCHS=30)。
新增:
  --arch   r2plus1d18(基线) | mvitv2s(Kinetics-400 预训练, 输入 224x224)
  --bigru  R(2+1)D 骨干逐帧特征 -> BiGRU(hidden 256 x 2 层双向) -> fc (0=关)
  --seed   随机种子(multi-seed 判读用), checkpoint 内记录全部元信息供部署端集成
判读铁律: 配置间差异 <5pt = 噪声; multi-seed x3 完成前不谈任何模型差异。
"""
import os
import math
import json
import random
import time
import argparse

import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision
from torchvision.models.video import R2Plus1D_18_Weights, MViT_V2_S_Weights

CLASSES = ['normal', 'dribble', 'defense', 'shoot', 'pass']
EPOCHS = 30
PATIENCE = 8
WORKERS = 4


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


class ClipDS(Dataset):
    def __init__(self, root, split, train_aug, n=16, size=112,
                 mean=None, std=None, ball_map=None, ball_x=None):
        self.items = []
        for ci, c in enumerate(CLASSES):
            d = os.path.join(root, split, c)
            for name in sorted(os.listdir(d)):
                self.items.append((os.path.join(d, name), ci))
        self.aug = train_aug
        self.n = n
        self.size = size
        self.mean = mean.view(3, 1, 1, 1)
        self.std = std.view(3, 1, 1, 1)
        self.ball_map = ball_map                 # {clip名: npz 行号}
        self.ball_x = ball_x                     # (N,16,4) float32

    def __len__(self):
        return len(self.items)

    def _load(self, d):
        fr = []
        for j in range(self.n):
            img = cv2.imdecode(np.fromfile(os.path.join(d, f'f{j:02d}.jpg'),
                                           dtype=np.uint8), cv2.IMREAD_COLOR)
            fr.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return np.stack(fr)  # (n,112,112,3) uint8

    def _ball(self, d):
        """-> (16,4) 人球相对序列; 未覆盖(被丢弃)的 clip 补零=球不可见"""
        if self.ball_x is None:
            return np.zeros((self.n, 4), np.float32)
        idx = self.ball_map.get(os.path.basename(d))
        if idx is None:
            return np.zeros((self.n, 4), np.float32)
        return self.ball_x[idx]

    def __getitem__(self, i):
        d, label = self.items[i]
        x = self._load(d)
        if self.aug:
            # 时序一致: 整 clip 同一几何变换
            if random.random() < 0.5:
                x = x[:, :, ::-1].copy()
            sc = random.uniform(0.7, 1.0)
            oh, ow = x.shape[1:3]
            ch, cw = int(oh * sc), int(ow * sc)
            y0 = random.randint(0, oh - ch)
            x0 = random.randint(0, ow - cw)
            x = x[:, y0:y0 + ch, x0:x0 + cw]
            x = np.stack([cv2.resize(f, (self.size, self.size)) for f in x])
            # 轻色彩抖动(整 clip 同参数)
            if random.random() < 0.5:
                b = random.uniform(0.8, 1.2)
                c = random.uniform(0.8, 1.2)
                x = np.clip((x.astype(np.float32) - 128) * c * b + 128,
                            0, 255).astype(np.uint8)
        elif self.size != x.shape[1]:
            x = np.stack([cv2.resize(f, (self.size, self.size)) for f in x])
        t = torch.from_numpy(x.copy()).permute(3, 0, 1, 2).float() / 255.0
        t = (t - self.mean) / self.std
        b = torch.from_numpy(self._ball(d).copy())
        return t, b, label


class R21DGRU(nn.Module):
    """R(2+1)D-18 骨干(去时间池化/fc) -> 逐帧空间池化特征序列 (B,T,512)
    -> 2 层 BiGRU(hidden 256) -> 时间均值 -> fc。骨干权重来自 Kinetics-400 预训练。"""

    def __init__(self, m, fc_in, num_classes, hidden=256, layers=2, dropout=0.3):
        super().__init__()
        self.stem = m.stem
        self.layer1, self.layer2 = m.layer1, m.layer2
        self.layer3, self.layer4 = m.layer3, m.layer4
        self.gru = nn.GRU(fc_in, hidden, num_layers=layers, batch_first=True,
                          bidirectional=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(2 * hidden, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)                       # (B,512,T,h,w)
        x = x.mean(dim=(3, 4)).transpose(1, 2)   # (B,T,512)
        y, _ = self.gru(x)
        return self.head(y.mean(dim=1))


def build_model(arch, bigru=0, ballrel=0):
    if arch == 'r2plus1d18':
        m = torchvision.models.video.r2plus1d_18(
            weights=R2Plus1D_18_Weights.KINETICS400_V1)
        if ballrel:
            return BallFused(R21DFeat(m), m.fc.in_features)
        if bigru > 0:
            return R21DGRU(m, m.fc.in_features, len(CLASSES))
        m.fc = nn.Linear(m.fc.in_features, len(CLASSES))
        return m
    if arch == 'mvitv2s':
        m = torchvision.models.video.mvit_v2_s(
            weights=MViT_V2_S_Weights.KINETICS400_V1)
        hin = m.head[-1].in_features if isinstance(m.head, (nn.Sequential,)) \
            else m.head.in_features
        if ballrel:
            if isinstance(m.head, nn.Sequential):
                m.head[-1] = nn.Identity()     # 保留 dropout/permute/norm, 输出 (B,hin)
            else:
                m.head = nn.Identity()
            return BallFused(m, hin)
        m.head = nn.Linear(hin, len(CLASSES))
        return m
    raise ValueError(arch)


class R21DFeat(nn.Module):
    """R(2+1)D-18 骨干 -> 全局池化特征 (B,512), 供 BallFused 外观分支。"""

    def __init__(self, m):
        super().__init__()
        self.stem = m.stem
        self.layer1, self.layer2 = m.layer1, m.layer2
        self.layer3, self.layer4 = m.layer3, m.layer4
        self.pool = m.avgpool

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.pool(x).flatten(1)           # (B,512)


class BallFused(nn.Module):
    """双分支联合训练: 外观骨干全局特征 + 人球相对位置 BiGRU 分支(16x4: dx/dy/dist/vis)
    特征级 concat -> 单头单损失(规避融合稀释定律)。
    上界实验: 纯人球 GRU 单独 clip macroF1=0.678 (runs/cls2/ballhead_eval.json)。"""

    def __init__(self, backbone, feat_dim, hidden=64, layers=2, dropout=0.3):
        super().__init__()
        self.backbone = backbone
        self.gru = nn.GRU(4, hidden, num_layers=layers, batch_first=True,
                          bidirectional=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(feat_dim + 2 * hidden, len(CLASSES))

    def forward(self, x, brel=None):
        a = self.backbone(x)
        y, _ = self.gru(brel)
        return self.head(torch.cat([a, y.mean(dim=1)], 1))


NORM = {
    'r2plus1d18': (torch.tensor([0.43216, 0.394666, 0.37645]),
                   torch.tensor([0.22803, 0.22145, 0.216989])),
    'mvitv2s': (torch.tensor([0.45, 0.45, 0.45]),
                torch.tensor([0.225, 0.225, 0.225])),
}


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    cm = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    for x, b, y in loader:
        if isinstance(model, BallFused):
            out = model(x.to(dev), b.to(dev))
        else:
            out = model(x.to(dev))
        p = out.argmax(1).cpu().numpy()
        for a, b2 in zip(y.numpy(), p):
            cm[a, b2] += 1
    f1s = []
    for i in range(len(CLASSES)):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return cm, float(np.mean(f1s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root',
                    default=r'F:\zjl\basket\二阶段起步\datasets\dataset_action_clip_val35_sliding')
    ap.add_argument('--run', default=r'F:\zjl\basket\二阶段2.0\runs\cls2')
    ap.add_argument('--tag', required=True)
    ap.add_argument('--arch', default='r2plus1d18',
                    choices=['r2plus1d18', 'mvitv2s'])
    ap.add_argument('--bigru', type=int, default=0,
                    help='>0 时启用 R(2+1)D+BiGRU 头(值仅为开关标志, hidden 固定 256x2)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--batch', type=int, default=0, help='0=按架构自动(16/8)')
    ap.add_argument('--epochs', type=int, default=EPOCHS)
    ap.add_argument('--patience', type=int, default=PATIENCE,
                    help='0=禁用早停(用户 2026-10-29 指示: 跑满 epochs)')
    ap.add_argument('--ball-rel', type=int, default=0, dest='ball_rel',
                    help='>0 启用人球相对位置分支(BallFused 双分支联合训练)')
    ap.add_argument('--ball-npz', dest='ball_npz',
                    default=r'F:\zjl\basket\二阶段2.0\runs\cls2\ballrel_data.npz')
    args = ap.parse_args()

    size = 224 if args.arch == 'mvitv2s' else 112
    batch = args.batch or (8 if args.arch == 'mvitv2s' else 16)
    RUN = os.path.join(args.run, args.tag)
    os.makedirs(RUN, exist_ok=True)

    set_seed(args.seed)
    dev = 'cuda'
    mean, std = NORM[args.arch]

    ball_map_t = ball_map_v = None
    bx_t = bx_v = None
    if args.ball_rel:
        d = np.load(args.ball_npz, allow_pickle=True)
        bx_t, bx_v = d['train_x'], d['val_x']
        # 按 clip 名对齐(npz 构建时丢弃过无人框 clip, 索引不齐, 名字齐)
        ball_map_t = {os.path.basename(m[3]): i
                      for i, m in enumerate(d['train_meta'])}
        ball_map_v = {os.path.basename(m[3]): i
                      for i, m in enumerate(d['val_meta'])}
        print(f'ballrel npz: train={len(bx_t)} val={len(bx_v)}')

    tr = ClipDS(args.data_root, 'train', True, n=16, size=size, mean=mean,
                std=std, ball_map=ball_map_t, ball_x=bx_t)
    va = ClipDS(args.data_root, 'val', False, n=16, size=size, mean=mean,
                std=std, ball_map=ball_map_v, ball_x=bx_v)
    if args.ball_rel:
        cov_t = sum(1 for pth, _ in tr.items
                    if os.path.basename(pth) in ball_map_t) / len(tr.items)
        cov_v = sum(1 for pth, _ in va.items
                    if os.path.basename(pth) in ball_map_v) / len(va.items)
        print(f'ballrel 覆盖率: train={cov_t:.3f} val={cov_v:.3f} '
              f'(未覆盖 clip 全零向量=球不可见)', flush=True)
    cnt = np.bincount([y for _, y in tr.items], minlength=5).astype(float)
    w = [(1.0 / math.sqrt(cnt[y])) for _, y in tr.items]      # sqrt 采样(1.0 基线同源)
    sampler = WeightedRandomSampler(w, num_samples=len(tr), replacement=True)
    trl = DataLoader(tr, batch, sampler=sampler, num_workers=WORKERS,
                     pin_memory=True, persistent_workers=True)
    val = DataLoader(va, batch, shuffle=False, num_workers=WORKERS,
                     pin_memory=True, persistent_workers=True)

    model = build_model(args.arch, args.bigru, args.ball_rel).to(dev)
    n_par = sum(p.numel() for p in model.parameters()) / 1e6
    head_mod = getattr(model, 'head', None) or model.fc    # R21DGRU/BallFused/MViT=head, 纯R21D=fc
    last = head_mod[-1] if isinstance(head_mod, nn.Sequential) else head_mod
    head_par = sum(p.numel() for p in last.parameters()) / 1e6
    if args.arch == 'r2plus1d18' and args.bigru:
        head_par += sum(p.numel() for p in model.gru.parameters()) / 1e6
    if args.ball_rel:
        head_par += sum(p.numel() for p in model.gru.parameters()) / 1e6
    print(f'arch={args.arch} bigru={args.bigru} ballrel={args.ball_rel} '
          f'seed={args.seed} lr={args.lr} '
          f'batch={batch} size={size} train={len(tr)} val={len(va)} '
          f'参数={n_par:.1f}M(头/新增={head_par:.1f}M)', flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    amp = torch.amp.GradScaler('cuda')

    best_f1, bad = 0.0, 0
    log_path = os.path.join(RUN, 'train_log.txt')
    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        tl = 0.0
        nb = 0
        for x, b, y in trl:
            x = x.to(dev, non_blocking=True)
            b = b.to(dev, non_blocking=True)
            y = y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                if isinstance(model, BallFused):
                    loss = crit(model(x, b), y)
                else:
                    loss = crit(model(x), y)
            amp.scale(loss).backward()
            amp.step(opt)
            amp.update()
            tl += loss.item()
            nb += 1
        sch.step()
        cm, mf1 = evaluate(model, val, dev)
        acc = cm.trace() / cm.sum()
        line = (f'ep{ep:03d} loss={tl/max(nb,1):.4f} val_acc={acc:.3f} '
                f'macroF1={mf1:.3f} time={time.time()-t0:.0f}s')
        print(line, flush=True)
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(line + '\n')
        if mf1 > best_f1:
            best_f1, bad = mf1, 0
            torch.save({'model': model.state_dict(), 'classes': CLASSES,
                        'arch': args.arch, 'bigru': args.bigru, 'size': size,
                        'ball_rel': args.ball_rel,
                        'seed': args.seed, 'lr': args.lr,
                        'val_macroF1': mf1, 'epoch': ep},
                       os.path.join(RUN, 'best.pt'))
            with open(os.path.join(RUN, 'confusion_best.json'), 'w',
                      encoding='utf-8') as f:
                json.dump({'classes': CLASSES, 'cm': cm.tolist(),
                           'macroF1': mf1, 'epoch': ep, 'arch': args.arch,
                           'bigru': args.bigru, 'ball_rel': args.ball_rel,
                           'seed': args.seed},
                          f, ensure_ascii=False, indent=1)
            print('  -> 保存 best', flush=True)
        else:
            bad += 1
            if args.patience > 0 and bad >= args.patience:
                print(f'早停于 ep{ep} (best ep={ep-bad}, macroF1={best_f1:.3f})',
                      flush=True)
                break

    ck = torch.load(os.path.join(RUN, 'best.pt'), map_location=dev)
    model.load_state_dict(ck['model'])
    cm, mf1 = evaluate(model, val, dev)
    print(f'\n最终 best(ep{ck["epoch"]}) macroF1={mf1:.3f}')
    print('混淆矩阵 (行=真实, 列=预测):')
    print('             ' + ' '.join(f'{c:>9s}' for c in CLASSES))
    for i, c in enumerate(CLASSES):
        print(f'{c:>11s}  ' + ' '.join(f'{cm[i, j]:9d}' for j in range(5)))
    for i, c in enumerate(CLASSES):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        p = tp / (tp + fp) if tp + fp else 0
        r = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * p * r / (p + r) if p + r else 0
        print(f'  {c:8s} P={p:.3f} R={r:.3f} F1={f1:.3f} 支持={int(cm[i, :].sum())}')


if __name__ == '__main__':
    main()
