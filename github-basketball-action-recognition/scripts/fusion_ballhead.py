# -*- coding: utf-8 -*-
"""两流融合评估: 外观集成概率 x 人球GRU概率 -> 网格搜权重 -> macroF1。
用法: yolo 环境 python fusion_ballhead.py
依赖: runs/cls2/{val_probs_ensemble,val_probs_ball}.npy + val_probs_labels.npy
"""
import os

os.environ['PYTHONUTF8'] = '1'

import json
import numpy as np

CLASSES = ['normal', 'dribble', 'defense', 'shoot', 'pass']
RUN = r'F:\zjl\basket\二阶段2.0\runs\cls2'


def macro_f1(y, pred):
    f1s = []
    for i in range(5):
        tp = int(((y == i) & (pred == i)).sum())
        fp = int(((y != i) & (pred == i)).sum())
        fn = int(((y == i) & (pred != i)).sum())
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return float(np.mean(f1s)), f1s


def main():
    pe = np.load(os.path.join(RUN, 'val_probs_ensemble.npy'))
    pb = np.load(os.path.join(RUN, 'val_probs_ball.npy'))
    y = np.load(os.path.join(RUN, 'val_probs_labels.npy'))
    assert len(pe) == len(pb) == len(y)
    n = min(len(pe), len(pb))          # 保险: 顺序一致 (同一 loader 口径)
    pe, pb, y = pe[:n], pb[:n], y[:n]
    print(f'clips={n}')

    m_ens, f1e = macro_f1(y, pe.argmax(1))
    m_ball, f1b = macro_f1(y, pb.argmax(1))
    print(f'单流 外观集成: macroF1={m_ens:.3f}  '
          + ' '.join(f'{c}={f:.3f}' for c, f in zip(CLASSES, f1e)))
    print(f'单流 人球GRU : macroF1={m_ball:.3f}  '
          + ' '.join(f'{c}={f:.3f}' for c, f in zip(CLASSES, f1b)))

    best = (0.0, None)
    print('\n权重扫描 w*p_ens+(1-w)*p_ball:')
    for w in np.arange(0.0, 1.01, 0.05):
        mf1, _ = macro_f1(y, (w * pe + (1 - w) * pb).argmax(1))
        mark = ''
        if mf1 > best[0]:
            best = (mf1, w)
            mark = ' <-'
        print(f'  w={w:.2f} macroF1={mf1:.3f}{mark}')
    mf1, f1s = macro_f1(y, (best[1] * pe + (1 - best[1]) * pb).argmax(1))
    cm = np.zeros((5, 5), int)
    pred = (best[1] * pe + (1 - best[1]) * pb).argmax(1)
    for a, b in zip(y, pred):
        cm[a, b] += 1
    print(f'\n=== 融合最优 w={best[1]:.2f} macroF1={mf1:.3f} ===')
    print('            ' + ' '.join(f'{c:>9s}' for c in CLASSES))
    for i, c in enumerate(CLASSES):
        print(f'{c:>11s} ' + ' '.join(f'{cm[i, j]:9d}' for j in range(5)))
    print('每类 F1: ' + ' '.join(f'{c}={f:.3f}'
                                  for c, f in zip(CLASSES, f1s)))
    with open(os.path.join(RUN, 'fusion_eval.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'macroF1_ens': m_ens, 'macroF1_ball': m_ball,
                   'macroF1_fused': mf1, 'best_w': float(best[1]),
                   'per_class_fused': f1s, 'cm': cm.tolist()},
                  f, ensure_ascii=False, indent=1)
    print('saved fusion_eval.json')


if __name__ == '__main__':
    main()
