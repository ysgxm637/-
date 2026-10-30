# -*- coding: utf-8 -*-
"""任务2汇总: 三档球输入条件融合指标 + 帧段对照 -> 02_ball_input 产物。"""
import os
import csv
import json
import numpy as np

RUN = r'F:\zjl\basket\二阶段2.0\runs\cls2'
OUT = r'f:\zjl\basket\四论文融合\experiments\v3_fusion_2_1_20261030\02_ball_input'
os.makedirs(OUT, exist_ok=True)
CLASSES = ['normal', 'dribble', 'defense', 'shoot', 'pass']

pe = np.load(os.path.join(RUN, 'val_probs_ensemble.npy'))
y = np.load(os.path.join(RUN, 'val_probs_labels.npy'))
d = np.load(os.path.join(RUN, 'ballrel_data.npz'), allow_pickle=True)
meta = [tuple(x) for x in d['val_meta']]
starts = np.array([m[2] for m in meta])


def mf1(pred, yy=None):
    yy = y if yy is None else yy
    f1s = []
    for i in range(5):
        tp = int(((yy == i) & (pred == i)).sum())
        fp = int(((yy != i) & (pred == i)).sum())
        fn = int(((yy == i) & (pred != i)).sum())
        pr = tp / (tp + fp) if tp + fp else 0
        rc = tp / (tp + fn) if tp + fn else 0
        f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0)
    return float(np.mean(f1s)), f1s


def seg_of(s):
    return '0-400' if s < 400 else ('400-1000' if s < 1000 else '1000-1235')


segs = np.array([seg_of(s) for s in starts])
rows = []
for cond in ('gt', 'det', 'det_nodeintp'):
    pb = np.load(os.path.join(RUN, f'val_probs_ball_{cond}.npy'))
    ms, _ = mf1(pb.argmax(1))
    mf, ff = mf1((0.30 * pe + 0.70 * pb).argmax(1))
    rows.append(dict(cond=cond, ball_single=round(ms, 3), fused=round(mf, 3),
                     per_class={c: round(f, 3) for c, f in zip(CLASSES, ff)}))
    print(cond, 'single', round(ms, 3), 'fused', round(mf, 3),
          rows[-1]['per_class'], flush=True)

pb = np.load(os.path.join(RUN, 'val_probs_ball_det.npy'))
pf = (0.30 * pe + 0.70 * pb).argmax(1)
pa = pe.argmax(1)
seg_rows = []
for sg in ('0-400', '400-1000', '1000-1235'):
    m = segs == sg
    ma, _ = mf1(pa[m], y[m])
    mf, _ = mf1(pf[m], y[m])
    seg_rows.append(dict(seg=sg, n=int(m.sum()), appearance=round(ma, 3),
                         fused_det=round(mf, 3)))
    print('seg', sg, 'n', int(m.sum()), 'appearance', round(ma, 3),
          'fused_det', round(mf, 3), flush=True)

with open(os.path.join(OUT, 'input_condition_metrics.json'), 'w',
          encoding='utf-8') as f:
    json.dump({'conditions': rows, 'segments_det_fused': seg_rows}, f,
              ensure_ascii=False, indent=1)
with open(os.path.join(OUT, 'input_condition_metrics.csv'), 'w', newline='',
          encoding='utf-8-sig') as f:
    w = csv.writer(f)
    w.writerow(['cond', 'ball_single_macroF1', 'fused_macroF1']
               + ['F1_' + c for c in CLASSES])
    for r in rows:
        w.writerow([r['cond'], r['ball_single'], r['fused']]
                   + [r['per_class'][c] for c in CLASSES])
    for r in seg_rows:
        w.writerow(['seg@{}(n={})'.format(r['seg'], r['n']), '',
                    r['fused_det']] + [r['appearance']])
print('saved 02_ball_input metrics')
