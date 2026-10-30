# -*- coding: utf-8 -*-
"""人球相对位置序列数据集构建 (上界实验: 全部来自人工 GT 标注)。
对 dataset_action_clip_val35_sliding 的每个 clip {vid}_t{gtid}_s{start}:
  16 帧 = start + j*2 @60fps
  特征/帧: dx=(bx-cx)/w, dy=(by-cy)/h, dist=hypot(dx,dy), vis(球可见)
  球中心来自 CVAT ball track (人工), 人框来自 player track (人工)
输出: runs/cls2/ballrel_data.npz (train/val x/y + clip 索引)
"""
import os
import re
import xml.etree.ElementTree as ET

import numpy as np

DATA = r'F:\zjl\basket\二阶段起步\datasets\dataset_action_clip_val35_sliding'
XML_DIR = r'F:\zjl\basket\2版'
OUT = r'F:\zjl\basket\二阶段2.0\runs\cls2\ballrel_data.npz'
CLASSES = ['normal', 'dribble', 'defense', 'shoot', 'pass']
CLIP_LEN, STRIDE = 16, 2
NAME_RE = re.compile(r'^(\d+)_t(\d+)_s(\d+)$')


def load_tracks(vid):
    """-> player {idx: {frame: bbox}}, ball {frame: (bx,by)}"""
    root = ET.parse(os.path.join(XML_DIR, f'{vid}.xml')).getroot()
    players, balls = {}, {}
    for idx, tr in enumerate(root.findall('track')):
        lab = tr.get('label')
        for b in tr.findall('box'):
            if b.get('outside') == '1':
                continue
            fr = int(b.get('frame'))
            xtl, ytl = float(b.get('xtl')), float(b.get('ytl'))
            xbr, ybr = float(b.get('xbr')), float(b.get('ybr'))
            if lab == 'player':
                players.setdefault(idx, {})[fr] = (xtl, ytl, xbr, ybr)
            elif lab == 'ball':
                balls[fr] = ((xtl + xbr) / 2, (ytl + ybr) / 2)
    return players, balls


def main():
    # 扫描 clip 目录
    clips = {'train': [], 'val': []}
    for split in clips:
        for c in CLASSES:
            d = os.path.join(DATA, split, c)
            for name in sorted(os.listdir(d)):
                m = NAME_RE.match(name)
                assert m, name
                clips[split].append(
                    (int(m.group(1)), int(m.group(2)), int(m.group(3)),
                     CLASSES.index(c), os.path.join(d, name)))
    print(f'train={len(clips["train"])} val={len(clips["val"])}')

    # 修复: 验证 gtid 与 XML track idx 对应 (抽 2 个 clip 目测帧框比例)
    cache = {}
    xs = {'train': [], 'val': []}
    ys = {'train': [], 'val': []}
    meta = {'train': [], 'val': []}
    n_noball = n_noply = 0
    for split in ('train', 'val'):
        for vid, gtid, start, label, dpath in clips[split]:
            if vid not in cache:
                cache[vid] = load_tracks(vid)
            players, balls = cache[vid]
            seq = np.zeros((CLIP_LEN, 4), np.float32)
            ok = True
            for j in range(CLIP_LEN):
                fr = start + j * STRIDE
                bb = players.get(gtid, {}).get(fr)
                if bb is None:            # 人框缺 -> 邻近帧补
                    near = sorted(players.get(gtid, {}),
                                  key=lambda f: abs(f - fr))
                    if not near:
                        ok = False
                        break
                    bb = players[gtid][near[0]]
                cx, cy = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2
                w = max(bb[2] - bb[0], 4.0)
                h = max(bb[3] - bb[1], 4.0)
                bpt = balls.get(fr)
                if bpt is None:
                    seq[j] = (0.0, 0.0, 0.0, 0.0)
                    n_noball += 1
                else:
                    dx = (bpt[0] - cx) / w
                    dy = (bpt[1] - cy) / h
                    seq[j] = (dx, dy, float(np.hypot(dx, dy)), 1.0)
            if not ok:
                n_noply += 1
                continue
            xs[split].append(seq)
            ys[split].append(label)
            meta[split].append((vid, gtid, start, dpath))
    for split in ('train', 'val'):
        xs[split] = np.stack(xs[split])
        ys[split] = np.array(ys[split], np.int64)
    ball_vis = float(np.mean([xs[s][..., 3].mean() for s in ('train', 'val')]))
    print(f'ball 可见帧比例={ball_vis:.2f} 无人框丢弃 clip={n_noply}')
    np.savez_compressed(OUT,
                        train_x=xs['train'], train_y=ys['train'],
                        val_x=xs['val'], val_y=ys['val'],
                        train_meta=np.array(meta['train'], object),
                        val_meta=np.array(meta['val'], object))
    print(f'-> {OUT}  train_x={xs["train"].shape} val_x={xs["val"].shape}')


if __name__ == '__main__':
    main()
