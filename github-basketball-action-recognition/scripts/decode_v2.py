# -*- coding: utf-8 -*-
"""离线解码实验(2.0 阶梯①②): 读 probs_N.csv(解码前全类概率, 由 deploy_action_e2e.py
--dump-probs 产生) + boxes_N.csv, 离线实现三种解码器, 输出与管线同格式的
timeline_N.csv 到 deploy_e2e_<tag>/, 供 evaluate_v2.py --runs-root 统一口径打分。

解码器(tag 语法):
  am0                        argmax 基线复现(自校验: 应与源 run 的 timeline 逐位一致)
  pk_t<θo×100>[_d<D>][_f<floor×100>][_h<hold×100>][_r<R>][_l<L>]
                             阶梯① 峰值检测+时间NMS(PES 思想):
                               - 仅对稀有类(dribble/defense/shoot/pass)建概率曲线
                               - 峰=±R 源帧内局部极大且 p>=θo; 按峰高降序做同类时间NMS(间距>D)
                               - 支撑域: 从峰向两侧扩到 p<hold, 上限 ±L 源帧
                               - 标签: 落入支撑域 -> 该类(多类取概率高者);
                                 否则 argmax, 但 argmax 为稀有类且 p<floor 时降级 normal
                               默认 hold=θo-0.2, R=8, D=20, L=30, floor=0.75 (源帧@60fps)
  hy_e<enter×100>_h<hold×100>[_m<margin×100>]
                             阶梯② 双阈值滞回(有状态):
                               - 稀有类进入需 p>=enter 且为 argmax; 进入后粘滞到 p<hold
                               - 在持期间他类超过当前 +margin 且 >=enter 可切换
                               默认 margin=0.25
force 语义与管线一致: force>=0 的步直接输出该类 one-hot(prob=0.90), 不参与曲线。

用法:
  python decode_v2.py --vids 3 5 --decoders am0 pk_t50 pk_t40 hy_e50_h30
"""
import os
import csv
import shutil
import argparse
from collections import defaultdict

RUNS_20 = r'F:\zjl\basket\二阶段2.0\runs'
CLASSES = ['normal', 'dribble', 'defense', 'shoot', 'pass']
RARE = CLASSES[1:]
NRARE = len(RARE)


def load_probs(vid, src, runs_root):
    d = os.path.join(runs_root, f'deploy_e2e_{src}')
    tl = defaultdict(list)
    with open(os.path.join(d, f'probs_{vid}.csv'), encoding='utf-8') as f:
        for row in csv.DictReader(f):
            pvec = [float(row[f'p_{c}']) for c in CLASSES]
            tl[int(row['track_id'])].append((int(row['frame']), pvec, int(row['force'])))
    for tid in tl:
        tl[tid].sort(key=lambda x: x[0])
    return tl


def decode_argmax(steps):
    out = []
    for f, pv, force in steps:
        if force >= 0:
            out.append((f, CLASSES[force], 0.90))
            continue
        j = max(range(len(pv)), key=lambda k: pv[k])
        out.append((f, CLASSES[j], pv[j]))
    return out


def peaks_of(curve, R, D, theta_open):
    """curve: {frame: p} -> 局部极大候选经同类时间NMS后 [(frame, p)] (按帧升序)"""
    fs = sorted(curve)
    cand = []
    for f in fs:
        p = curve[f]
        if p < theta_open:
            continue
        if all(curve[g] <= p for g in fs if f - R <= g <= f + R):
            cand.append((f, p))
    cand.sort(key=lambda x: (-x[1], x[0]))
    kept = []
    for f, p in cand:
        if all(abs(f - g) > D for g, _ in kept):
            kept.append((f, p))
    kept.sort()
    return kept


def support_of(fs, i_peak, curve, hold, L):
    """fs 升序帧表, i_peak 峰下标: 从峰向两侧扩到 p<hold 或超出 L 源帧 -> [fl, fr]"""
    fp = fs[i_peak]
    fl = fr = fp
    j = i_peak - 1
    while j >= 0 and fp - fs[j] <= L and curve[fs[j]] >= hold:
        fl = fs[j]
        j -= 1
    j = i_peak + 1
    while j < len(fs) and fs[j] - fp <= L and curve[fs[j]] >= hold:
        fr = fs[j]
        j += 1
    return fl, fr


def decode_peak(steps, theta_open, hold, R, D, L, floor):
    curves = {c: {} for c in RARE}
    for f, pv, force in steps:
        if force >= 0:
            continue
        for ci, c in enumerate(RARE, start=1):
            curves[c][f] = pv[ci]
    sup = {}
    for c in RARE:
        fs = sorted(curves[c])
        sup[c] = [support_of(fs, fs.index(f), curves[c], hold, L)
                  for f, _ in peaks_of(curves[c], R, D, theta_open)]
    out = []
    for f, pv, force in steps:
        if force >= 0:
            out.append((f, CLASSES[force], 0.90))
            continue
        best_c, best_p = None, -1.0
        for ci, c in enumerate(RARE, start=1):
            if pv[ci] > best_p and any(fl <= f <= fr for fl, fr in sup[c]):
                best_c, best_p = c, pv[ci]
        if best_c is not None:
            out.append((f, best_c, best_p))
            continue
        j = max(range(len(pv)), key=lambda k: pv[k])
        if CLASSES[j] in RARE and pv[j] < floor:
            out.append((f, 'normal', pv[0]))
        else:
            out.append((f, CLASSES[j], pv[j]))
    return out


def decode_hyst(steps, enter, hold, margin, only=None):
    """only: 应用滞回的类别下标集合(默认全部稀有类); 其余类纯 argmax 直通"""
    only = only if only is not None else set(range(1, len(CLASSES)))
    out, state = [], 0
    for f, pv, force in steps:
        if force >= 0:
            out.append((f, CLASSES[force], 0.90))
            state = force
            continue
        j = max(range(len(pv)), key=lambda k: pv[k])
        if state in only and pv[state] < hold:
            state = 0
        if state in only and state != 0:
            if j != state and pv[j] > pv[state] + margin and (pv[j] >= enter or j not in only):
                state = j
            lab = state
        else:
            if j in only:
                state = j if pv[j] >= enter else 0
            else:
                state = j
            lab = state
        out.append((f, CLASSES[lab], pv[lab]))
    return out


def parse_tag(tag):
    """-> (decoder_fn, kwargs)"""
    if tag == 'am0':
        return decode_argmax, {}
    if tag.startswith('pk_'):
        kw = dict(theta_open=0.5, hold=None, R=8, D=20, L=30, floor=0.75)
        for tok in tag.split('_')[1:]:
            if tok.startswith('t'):
                kw['theta_open'] = int(tok[1:]) / 100
            elif tok.startswith('d'):
                kw['D'] = int(tok[1:])
            elif tok.startswith('f'):
                kw['floor'] = int(tok[1:]) / 100
            elif tok.startswith('h'):
                kw['hold'] = int(tok[1:]) / 100
            elif tok.startswith('r'):
                kw['R'] = int(tok[1:])
            elif tok.startswith('l'):
                kw['L'] = int(tok[1:])
        if kw['hold'] is None:
            kw['hold'] = round(kw['theta_open'] - 0.2, 2)
        return decode_peak, kw
    if tag.startswith(('hy_', 'hyd_')):
        kw = dict(enter=0.5, hold=0.3, margin=0.25)
        for tok in tag.split('_')[1:]:
            if tok.startswith('e'):
                kw['enter'] = int(tok[1:]) / 100
            elif tok.startswith('h'):
                kw['hold'] = int(tok[1:]) / 100
            elif tok.startswith('m'):
                kw['margin'] = int(tok[1:]) / 100
        if tag.startswith('hyd_'):                    # 仅 defense 滞回, 其余类 argmax 直通
            kw['only'] = {CLASSES.index('defense')}
        return decode_hyst, kw
    raise ValueError(f'未知解码器 tag: {tag}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--vids', nargs='+', default=['3', '5'])
    ap.add_argument('--decoders', nargs='+', required=True,
                    help='am0 / pk_* / hy_* (见文件头)')
    ap.add_argument('--src', default='probs0', help='概率来源 run tag(含 probs_N.csv)')
    ap.add_argument('--runs-root', dest='runs_root', default=RUNS_20)
    ap.add_argument('--prefix', default='',
                    help='输出 tag 前缀(避免不同 src 的同理解码结果互撞, 如 src=mvitd_s0 时用 m)')
    args = ap.parse_args()

    for vid in args.vids:
        probs = load_probs(vid, args.src, args.runs_root)
        n_tracks = len(probs)
        n_steps = sum(len(v) for v in probs.values())
        n_force = sum(1 for v in probs.values() for _, _, fo in v if fo >= 0)
        print(f'视频{vid}: 源={args.src} 轨迹={n_tracks} 预测步={n_steps} 强制步={n_force}')
        for tag in args.decoders:
            fn, kw = parse_tag(tag)
            out_tag = args.prefix + tag
            out_dir = os.path.join(args.runs_root, f'deploy_e2e_{out_tag}')
            os.makedirs(out_dir, exist_ok=True)
            rows = []
            for tid in sorted(probs):
                for f, lab, p in fn(probs[tid], **kw):
                    rows.append((f, tid, lab, p))
            rows.sort()
            csv_p = os.path.join(out_dir, f'timeline_{vid}.csv')
            with open(csv_p, 'w', newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                w.writerow(['frame', 'track_id', 'action_smooth', 'prob'])
                w.writerows(rows)
            shutil.copy(os.path.join(args.runs_root, f'deploy_e2e_{args.src}',
                                     f'boxes_{vid}.csv'),
                        os.path.join(out_dir, f'boxes_{vid}.csv'))
            n_ev = sum(1 for i, r in enumerate(rows)
                       if i == 0 or r[1] != rows[i - 1][1] or r[2] != rows[i - 1][2])
            print(f'  {tag}: {len(rows)}行 段变更~{n_ev} -> {csv_p}')
    print('完成。打分: python evaluate_v2.py --tags ... --runs-root', args.runs_root,
          '--label decode')


if __name__ == '__main__':
    main()
