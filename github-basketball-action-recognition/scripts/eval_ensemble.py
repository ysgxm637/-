# -*- coding: utf-8 -*-
"""四权重集成 val 评测: r21d_s0/s1/s2 + mvit_s2 概率平均 -> macroF1/混淆矩阵。
用法: yolo 环境 python eval_ensemble.py
数据: dataset_action_clip_val35_sliding/val (人工标注, 视频3/5, 6824 clip)
零 C 盘写入 (缓存走 F:\\zjl\\dltmp)。
"""
import os

# Zero C-drive writes
os.environ['TORCH_HOME'] = r'F:\zjl\dltmp\torch_home'
os.environ['HF_HOME'] = r'F:\zjl\dltmp\hf_home'
os.environ['MPLCONFIGDIR'] = r'F:\zjl\dltmp\mpl'
os.environ['TMP'] = r'F:\zjl\dltmp\tmp'
os.environ['TEMP'] = r'F:\zjl\dltmp\tmp'
os.environ['PYTHONUTF8'] = '1'

import json
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision

CLASSES = ['normal', 'dribble', 'defense', 'shoot', 'pass']
DATA = r'F:\zjl\basket\二阶段起步\datasets\dataset_action_clip_val35_sliding'
RUN = r'F:\zjl\basket\二阶段2.0\runs\cls2'
CKPTS = [os.path.join(RUN, t, 'best.pt')
         for t in ('r21d_s0', 'r21d_s1', 'r21d_s2', 'mvit_s2')]
NORM = {
    'r2plus1d18': (np.array([0.43216, 0.394666, 0.37645], np.float32),
                   np.array([0.22803, 0.22145, 0.216989], np.float32)),
    'mvitv2s': (np.array([0.45] * 3, np.float32),
                np.array([0.225] * 3, np.float32)),
}


class RawClips(Dataset):
    """val split, 只 load + resize 到统一 112, 返回 uint8 (16,112,112,3)"""

    def __init__(self, root, split='val', n=16, size=112):
        self.items = []
        for ci, c in enumerate(CLASSES):
            d = os.path.join(root, split, c)
            for name in sorted(os.listdir(d)):
                self.items.append((os.path.join(d, name), ci))
        self.n, self.size = n, size

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        d, label = self.items[i]
        fr = []
        for j in range(self.n):
            img = cv2.imdecode(np.fromfile(os.path.join(d, f'f{j:02d}.jpg'),
                                           dtype=np.uint8), cv2.IMREAD_COLOR)
            fr.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        x = np.stack(fr)
        if self.size != x.shape[1]:
            x = np.stack([cv2.resize(f, (self.size, self.size)) for f in x])
        return x, label


def build_model(arch):
    if arch == 'r2plus1d18':
        m = torchvision.models.video.r2plus1d_18(weights=None)
        m.fc = torch.nn.Linear(m.fc.in_features, len(CLASSES))
    else:
        m = torchvision.models.video.mvit_v2_s(weights=None)
        hin = m.head[-1].in_features if isinstance(m.head, (torch.nn.Sequential,)) \
            else m.head.in_features
        m.head = torch.nn.Linear(hin, len(CLASSES))
    return m


@torch.no_grad()
def main():
    import sys
    batch = int(sys.argv[1]) if len(sys.argv) > 1 else 8   # 显存紧张时用 2
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    models = []
    for p in CKPTS:
        ck = torch.load(p, map_location=dev, weights_only=False)
        arch = ck.get('arch', 'r2plus1d18')
        m = build_model(arch).to(dev)
        m.load_state_dict(ck['model'])
        m.eval()
        models.append((m, arch, ck.get('val_macroF1'), os.path.basename(
            os.path.dirname(p))))
        print(f'loaded {arch} val_macroF1={ck.get("val_macroF1")}', flush=True)

    ds = RawClips(DATA)
    loader = DataLoader(ds, batch, shuffle=False, num_workers=4, pin_memory=True)
    print(f'val clips: {len(ds)}', flush=True)

    probs_sum = np.zeros((len(ds), len(CLASSES)), np.float32)
    labels = np.zeros(len(ds), np.int64)
    i0 = 0
    for x, y in loader:                     # x uint8 (B,16,112,112,3)
        b = len(x)
        for m, arch, *_ in models:
            size = 224 if arch == 'mvitv2s' else 112
            if size != 112:                 # 112 存储帧 -> 双线性放大 224
                xf = x.view(-1, 112, 112, 3).permute(0, 3, 1, 2).float() / 255.0
                xf = F.interpolate(xf, size=(size, size), mode='bilinear',
                                   align_corners=False)
                xf = xf.view(b, 16, 3, size, size).permute(0, 2, 1, 3, 4)
            else:
                xf = x.permute(0, 4, 1, 2, 3).float() / 255.0   # (B,3,16,H,W)
            mean, std = NORM[arch]
            mt = torch.from_numpy(mean).to(dev).view(1, 3, 1, 1, 1)
            st = torch.from_numpy(std).to(dev).view(1, 3, 1, 1, 1)
            xx = (xf.to(dev) - mt) / st
            step = 8 if size == 224 else len(xx)
            with torch.amp.autocast('cuda'):
                p = torch.cat([torch.softmax(m(xx[i:i + step]), dim=1)
                               for i in range(0, len(xx), step)])
            probs_sum[i0:i0 + b] += p.float().cpu().numpy()
        labels[i0:i0 + b] = y.numpy()
        i0 += b
        if i0 % 800 < 8:
            print(f'{i0}/{len(ds)}', flush=True)

    np.save(os.path.join(RUN, 'val_probs_ensemble.npy'), probs_sum)
    np.save(os.path.join(RUN, 'val_probs_labels.npy'), labels)
    pred = probs_sum.argmax(1)
    cm = np.zeros((5, 5), int)
    for a, b in zip(labels, pred):
        cm[a, b] += 1
    f1s = []
    print('\n混淆矩阵 (行=真实, 列=预测):')
    print('            ' + ' '.join(f'{c:>9s}' for c in CLASSES))
    for i, c in enumerate(CLASSES):
        print(f'{c:>11s} ' + ' '.join(f'{cm[i, j]:9d}' for j in range(5)))
    print('\n每类 P/R/F1:')
    for i, c in enumerate(CLASSES):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        pr = tp / (tp + fp) if tp + fp else 0
        rc = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0
        f1s.append(f1)
        print(f'  {c:8s} P={pr:.3f} R={rc:.3f} F1={f1:.3f} 支持={int(cm[i].sum())}')
    mf1 = float(np.mean(f1s))
    acc = float(cm.trace() / cm.sum())
    print(f'\n=== 集成({len(models)}权重) macroF1={mf1:.3f} acc={acc:.3f} ===')
    for m, arch, f1, tag in models:
        print(f'  单模对照 {tag}({arch}): {f1}')

    with open(os.path.join(RUN, 'ensemble_eval.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'classes': CLASSES, 'cm': cm.tolist(), 'macroF1': mf1,
                   'acc': acc, 'per_class_f1': f1s,
                   'ckpts': [os.path.relpath(p, RUN) for p in CKPTS]},
                  f, ensure_ascii=False, indent=1)
    print('saved ensemble_eval.json')


if __name__ == '__main__':
    main()
