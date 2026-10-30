# -*- coding: utf-8 -*-
"""任务2: 三档球输入条件评估 (GT球 / 检测+插值球 / 仅检测帧)。
用已训 ballhead_best.pt 对 val6824 推理三档特征, 再与外观集成概率融合,
输出分段对照 (0-400 / 400-1000 / 1000-1235)。
用法: yolo 环境 python eval_ball_conditions.py
输出: 二阶段2.0/runs/cls2/ballcond_*.npy + 02_ball_input 指标(由下游脚本汇总)
"""
import os

os.environ['TORCH_HOME'] = r'F:\zjl\dltmp\torch_home'
os.environ['TMP'] = r'F:\zjl\dltmp\tmp'
os.environ['TEMP'] = r'F:\zjl\dltmp\tmp'
os.environ['PYTHONUTF8'] = '1'

import sys
import pickle
import xml.etree.ElementTree as ET
import numpy as np
import torch
import torch.nn as nn

CLASSES = ['normal', 'dribble', 'defense', 'shoot', 'pass']
RUN = r'F:\zjl\basket\二阶段2.0\runs\cls2'
S = r'f:\zjl\basket\四论文融合\states'
XML_DIR = r'F:\zjl\basket\2版'
DATA = r'F:\zjl\basket\二阶段起步\datasets\dataset_action_clip_val35_sliding'
CLIP_LEN, STRIDE = 16, 2


class BallGRU(nn.Module):
    def __init__(self, feat=4, hidden=64, layers=2, n_cls=5):
        super().__init__()
        self.gru = nn.GRU(feat, hidden, num_layers=layers, batch_first=True,
                          bidirectional=True, dropout=0.2)
        self.head = nn.Linear(2 * hidden, n_cls)

    def forward(self, x):
        y, _ = self.gru(x)
        return self.head(y.mean(dim=1))


def load_players_balls(vid):
    root = ET.parse(os.path.join(XML_DIR, f'{vid}.xml')).getroot()
    players, balls = {}, {}
    for tr in root.findall('track'):
        lab = tr.get('label')
        for b in tr.findall('box'):
            if b.get('outside') == '1':
                continue
            fr = int(b.get('frame'))
            xtl, ytl = float(b.get('xtl')), float(b.get('ytl'))
            xbr, ybr = float(b.get('xbr')), float(b.get('ybr'))
            if lab == 'player':
                players.setdefault(id(tr), {})[fr] = (xtl, ytl, xbr, ybr)
            elif lab == 'ball':
                balls[fr] = ((xtl + xbr) / 2, (ytl + ybr) / 2)
    # 重键: 用 track 在 XML 中的序号 (与 make_ballrel_data 的 gtid 对齐)
    players2 = {}
    for i, tr in enumerate(root.findall('track')):
        if tr.get('label') != 'player':
            continue
        d = {}
        for b in tr.findall('box'):
            if b.get('outside') == '1':
                continue
            fr = int(b.get('frame'))
            d[fr] = (float(b.get('xtl')), float(b.get('ytl')),
                     float(b.get('xbr')), float(b.get('ybr')))
        players2[i] = d
    del players
    return players2, balls


def build_feats(cond, val_meta, players_cache, det_cache, gt_cache):
    """cond: gt / det / det_nodeintp -> (N,16,4)"""
    xs = []
    for vid, gtid, start, dpath in val_meta:
        players = players_cache[vid]
        seq = np.zeros((CLIP_LEN, 4), np.float32)
        for j in range(CLIP_LEN):
            fr = start + j * STRIDE
            bb = players.get(gtid, {}).get(fr)
            if bb is None:
                near = sorted(players.get(gtid, {}), key=lambda f: abs(f - fr))
                bb = players[gtid][near[0]] if near else None
                if bb is None:
                    continue
            cx, cy = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2
            w = max(bb[2] - bb[0], 4.0)
            h = max(bb[3] - bb[1], 4.0)
            if cond == 'gt':
                pt = gt_cache.get(('gt', vid, fr))
            else:
                pt = det_cache.get(('det', vid, fr))
            if pt is None or (cond == 'det_nodeintp' and pt[2] == 1):
                seq[j] = 0.0
                continue
            dx = (pt[0] - cx) / w
            dy = (pt[1] - cy) / h
            seq[j] = (dx, dy, float(np.hypot(dx, dy)), 1.0)
        xs.append(seq)
    return np.stack(xs)


@torch.no_grad()
def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    d = np.load(os.path.join(RUN, 'ballrel_data.npz'), allow_pickle=True)
    val_meta = [tuple(x) for x in d['val_meta']]

    # 球源缓存
    gt_cache, det_cache = {}, {}
    players_cache = {}
    vids = sorted({m[0] for m in val_meta})
    for vid in vids:
        players, balls = load_players_balls(vid)
        players_cache[vid] = players
        for fr, pt in balls.items():
            gt_cache[('gt', vid, fr)] = (pt[0], pt[1], 0)
    # 检测球: balltraj_v3_dict.pkl (camel 预转换的纯 dict, 规避 yolo env 无 pandas)
    with open(os.path.join(RUN, 'balltraj_v3_dict.pkl'), 'rb') as f:
        BD = pickle.load(f)
    for fid, (bx, by, interp) in BD.items():
        if bx is not None:
            det_cache[('det', 3, fid)] = (bx, by, 1 if interp else 0)

    model = BallGRU().to(dev)
    ck = torch.load(os.path.join(RUN, 'ballhead_best.pt'), map_location=dev,
                    weights_only=False)
    model.load_state_dict(ck['model'])
    model.eval()

    from torch.utils.data import TensorDataset, DataLoader
    out = {}
    for cond in ('gt', 'det', 'det_nodeintp'):
        x = build_feats(cond, val_meta, players_cache, det_cache, gt_cache)
        loader = DataLoader(TensorDataset(torch.from_numpy(x)), 256,
                            shuffle=False)
        probs = np.zeros((len(x), 5), np.float32)
        i0 = 0
        for (xb,) in loader:
            p = torch.softmax(model(xb.to(dev)), dim=1).float().cpu().numpy()
            probs[i0:i0 + len(p)] = p
            i0 += len(p)
        np.save(os.path.join(RUN, f'val_probs_ball_{cond}.npy'), probs)
        vis = float(x[..., 3].mean())
        out[cond] = dict(vis_pct=round(vis, 3), probs=os.path.basename(
            f'val_probs_ball_{cond}.npy'))
        print(f'{cond}: vis={vis:.2f} -> val_probs_ball_{cond}.npy', flush=True)
    import json
    with open(os.path.join(RUN, 'ballcond_meta.json'), 'w',
              encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print('done')


def pickle_load(p):
    import pickle
    with open(p, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    main()
