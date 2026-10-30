# -*- coding: utf-8 -*-
"""二阶段 2.0 评测重装: 在 1.0 部署 CSV (timeline/boxes) 与 CVAT XML GT 之上重打分。

同时计算三套口径:
  [帧级]   macroF1 / microF1 / macroRecall(平衡准确率) / 端到端acc(漏检计错)
           + normal 截断平衡口径(normal 对采样到稀有类总量, FineGym Gym99 思路)
  [事件旧] tIoU>=0.3 贪心匹配 P/R/F1 —— 与 1.0 eval_deploy_gt.py 同源, 作对账锚点
  [事件新] mTAP@k 点容差 AP: 预测段与 GT 段同(视频,映射轨迹,类别)且
           预测触发点距 GT 事件 <= k 帧时为命中, 按置信度贪心一对一匹配(VOC 式 AP)。
           默认 --mode interval: 距离 = pred 锚点(起点)到 GT 区间 [s,e] 的距离
             (区间内=0) —— 段式 GT 的点容差正解, 治 tIoU 边界抖动放大。
           --mode anchor: 距离 = |pred锚点 - GT锚点| (VNL-STES 点事件原义, 更严)。
           容差默认 0/2/5/10 帧@60fps ≈ VNL mTAP@{0,1,2,4}@25fps。
           碎片化指数 = mTAP@0 - mTAP@max (MultiSports 双阈值思想)。
  [统计]   按事件重采样 bootstrap CI(事件级) + 按轨迹重采样 bootstrap CI(帧级 macroF1)

事件评估仅统计被映射轨迹覆盖的 GT(与 1.0 一致, 覆盖损失由端到端 acc 承担)。
事件匹配要求同轨迹(1.0 逐轨迹匹配的池化修正版, 避免 gtid 双计数)。
预测段分数 = 段内逐帧 prob 均值(timeline 的 prob 为该帧所选类别的解码置信度)。

用法:
  python evaluate_v2.py --tags t1 t2 [--videos 3 5] [--bootstrap 1000]
                        [--mode interval] [--anchor onset] [--tols 0 2 5 10]
输出: 每配置逐视频帧级明细 + 汇总表(stdout), 汇总 CSV 落盘 eval_v2 目录。
"""
import os
import sys
import csv
import random
import argparse
from collections import defaultdict, Counter

sys.path.insert(0, r'F:\zjl\basket\二阶段起步\scripts')
from build_action_clip_dataset import parse_video, CLASSES          # noqa: E402

XML_DIR = r'F:\zjl\basket\2版'
RUNS_10 = r'F:\zjl\basket\二阶段起步\runs'
OUT_DIR = r'F:\zjl\basket\二阶段2.0\runs\eval_v2'
RARE = [c for c in CLASSES if c != 'normal']


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    u = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / u if u > 0 else 0.0


def load_gt(vid, xml_dir=XML_DIR):
    """-> gt[gtid] = {frame: (bbox, action)}"""
    xml = os.path.join(xml_dir, f'{vid}.xml')
    gt = {}
    for ti, seq in parse_video(xml):
        gt[ti] = {fr: (bb, act) for fr, bb, act in seq}
    return gt


def load_pred(vid, tag, runs_root=RUNS_10):
    """-> pb[fi]=[(tid,box)], tl[tid]=[(frame,label,prob)] (按帧排序)"""
    d = os.path.join(runs_root, f'deploy_e2e_{tag}')
    pb = defaultdict(list)
    with open(os.path.join(d, f'boxes_{vid}.csv'), encoding='utf-8') as f:
        for row in csv.DictReader(f):
            pb[int(row['frame'])].append((int(row['track_id']),
                                          (float(row['x1']), float(row['y1']),
                                           float(row['x2']), float(row['y2']))))
    tl = defaultdict(list)
    with open(os.path.join(d, f'timeline_{vid}.csv'), encoding='utf-8') as f:
        for row in csv.DictReader(f):
            tl[int(row['track_id'])].append((int(row['frame']),
                                             row['action_smooth'], float(row['prob'])))
    for tid in tl:
        tl[tid].sort(key=lambda x: x[0])
    return pb, tl


def match_tracks(pb, gt, iou_th):
    """预测轨迹 tid -> GT 轨迹 gtid IoU 投票映射 (与 1.0 一致)"""
    votes = defaultdict(Counter)
    for fi, plist in pb.items():
        glist = [(gt_tid, d[fi][0]) for gt_tid, d in gt.items() if fi in d]
        if not glist:
            continue
        for tid, box in plist:
            best, bi = 0.0, None
            for gt_tid, gbox in glist:
                v = iou(box, gbox)
                if v > best:
                    best, bi = v, gt_tid
            if bi is not None and best >= iou_th:
                votes[tid][bi] += 1
    return {tid: c.most_common(1)[0][0] for tid, c in votes.items()}


def build_segments(lab_by_frame, min_gap=6):
    """逐帧标签网格上的连续同类段 -> [(start, end, label)] (与 1.0 一致)"""
    segs, cur = [], None
    for fi in sorted(lab_by_frame):
        a = lab_by_frame[fi]
        if cur is None:
            cur = [a, fi, fi]
        elif a != cur[0] or fi - cur[2] >= min_gap:
            segs.append((cur[1], cur[2], cur[0]))
            cur = [a, fi, fi]
        else:
            cur[2] = fi
    if cur is not None:
        segs.append((cur[1], cur[2], cur[0]))
    return segs


def forward_fill(preds):
    """timeline 前向填充 -> ({frame: label}, {frame: prob}), 与 1.0 同源(不设上限)"""
    lab, out = {}, {}
    for i, (pf, la, pr) in enumerate(preds):
        end = preds[i + 1][0] if i + 1 < len(preds) else pf + 1
        for f in range(pf, end):
            lab[f] = la
            out[f] = pr
    return lab, out


def seg_score(seg, prob_fill):
    """段分数 = 段内逐帧 prob 均值"""
    s, e, _ = seg
    vals = [prob_fill[f] for f in range(s, e + 1) if f in prob_fill]
    return sum(vals) / len(vals) if vals else 0.0


def cm_metrics(pairs):
    """pairs: [(gt, pred)] -> dict(macroF1, microF1, macroRec, acc)"""
    cm = defaultdict(lambda: defaultdict(int))
    for g, p in pairs:
        cm[g][p] += 1
    tp_all = fp_all = fn_all = 0
    f1s, n_cls, recs, n_rc = 0.0, 0, 0.0, 0
    for c in CLASSES:
        tp = cm[c][c]
        fp = sum(cm[g][c] for g in cm if g != c)
        fn = sum(cm[c][p] for p in cm[c] if p != c)
        p_ = tp / (tp + fp) if tp + fp else 0.0
        r_ = tp / (tp + fn) if tp + fn else 0.0
        if tp + fn:
            f1s += 2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0.0
            n_cls += 1
            recs += r_
            n_rc += 1
        tp_all += tp
        fp_all += fp
        fn_all += fn
    micro = tp_all / (tp_all + fp_all + fn_all) if tp_all + fp_all + fn_all else 0.0
    return dict(macroF1=f1s / max(n_cls, 1), microF1=micro,
                macroRec=recs / max(n_rc, 1), acc=tp_all / max(len(pairs), 1))


def truncated_pairs(pairs):
    """normal 截断平衡: normal 对采样到稀有类总量(全局等步长, 确定性)"""
    n_rare = sum(1 for g, *_ in pairs if g in RARE)
    n_normal = len(pairs) - n_rare
    if n_normal <= n_rare or n_rare == 0:
        return pairs
    stride = n_normal / n_rare
    out, k = [], 0
    for item in pairs:                      # pairs 已按 (vid,gtid,frame) 排序
        if item[0] != 'normal':
            out.append(item)
        else:
            k += 1
            if int(k * stride) != int((k - 1) * stride):
                out.append(item)
    return out


def make_dist_fn(mode, anchor_of):
    """-> dist(pred_event, gt_event): 命中判据用的帧距离"""
    if mode == 'interval':                  # pred 锚点到 GT 区间的距离(区间内=0)
        return lambda p, g: max(g['s'] - anchor_of(p), anchor_of(p) - g['e'], 0)
    return lambda p, g: abs(anchor_of(p) - anchor_of(g))   # anchor 对 anchor


def greedy_event_match(gt_ev, pr_ev, tol, dist_fn):
    """点容差贪心一对一: pred 按分数降序, 命中同(vid,gtid,cls)且 dist<=tol 的 GT。
    -> per-cls (tp, n_gt, n_pred, tp_rank集合, ps)"""
    res = {}
    for c in CLASSES:
        gs = gt_ev[c]
        ps = sorted(pr_ev[c], key=lambda e: (-e['score'], e['vid'], e['gtid'], e['s']))
        avail = list(range(len(gs)))
        tp_ranks = set()
        for pi, p in enumerate(ps):
            best, bg = None, None
            for gi in avail:
                g = gs[gi]
                if g['vid'] == p['vid'] and g['gtid'] == p['gtid']:
                    d = dist_fn(p, g)
                    if d <= tol and (best is None or d < best):
                        best, bg = d, gi
            if bg is not None:
                avail.remove(bg)
                tp_ranks.add(pi)
        res[c] = (len(tp_ranks), len(gs), len(ps), tp_ranks, ps)
    return res


def ap_from(tp_ranks, ps, n_gt):
    """VOC 式 AP: 每个 TP 位次的 precision 累加 / n_gt"""
    if n_gt == 0:
        return None
    hit, ap = 0, 0.0
    for i in range(len(ps)):
        if i in tp_ranks:
            hit += 1
            ap += hit / (i + 1)
    return ap / n_gt


def prf(tp, ng, npd):
    p = tp / npd if npd else 0.0
    r = tp / ng if ng else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def iou_t(tp_fp):
    """时间 IoU (闭区间)"""
    s1, e1, s2, e2 = tp_fp
    inter = min(e1, e2) - max(s1, s2) + 1
    if inter <= 0:
        return 0.0
    uni = max(e1, e2) - min(s1, s2) + 1
    return inter / uni


def legacy_event(gt_ev, pr_ev, tiou=0.3):
    """旧口径: 同(vid,gtid,cls)池内 tIoU 贪心匹配 -> per-cls (tp,ng,npred)"""
    out = {}
    for c in CLASSES:
        gs, ps = gt_ev[c], pr_ev[c]
        used, tp = set(), 0
        for g in gs:
            best, bj = 0.0, -1
            for pi, p in enumerate(ps):
                if pi in used or p['vid'] != g['vid'] or p['gtid'] != g['gtid']:
                    continue
                v = iou_t((g['s'], g['e'], p['s'], p['e']))
                if v > best:
                    best, bj = v, pi
            if bj >= 0 and best >= tiou:
                used.add(bj)
                tp += 1
        out[c] = (tp, len(gs), len(ps))
    return out


def event_metrics(gt_ev, pr_ev, tols, dist_fn):
    """-> (per-cls legacy dict, {tol: (microF1, macroF1, mTAP, percls)}, ng, npd)"""
    lg = legacy_event(gt_ev, pr_ev)
    ng = sum(len(gt_ev[c]) for c in CLASSES)
    npd = sum(len(pr_ev[c]) for c in CLASSES)
    out = {}
    for k in tols:
        m = greedy_event_match(gt_ev, pr_ev, k, dist_fn)
        tp = sum(m[c][0] for c in CLASSES)
        _, _, f1 = prf(tp, ng, npd)
        aps = [ap_from(m[c][3], m[c][4], m[c][1]) for c in CLASSES if m[c][1]]
        fs = [prf(*m[c][:3])[2] for c in CLASSES if m[c][1]]
        out[k] = (f1, sum(fs) / len(fs) if fs else 0.0,
                  sum(aps) / len(aps) if aps else 0.0, m)
    return lg, out, ng, npd


def bootstrap_event(gt_ev, pr_ev, tols, dist_fn, n_draw, rng):
    """按事件重采样(类内分层有放回) -> 各容差 microF1/macroF1/mTAP 与 tIoUF1 的 95% CI"""
    keys = ([f'F1@{k}' for k in tols] + [f'mF1@{k}' for k in tols]
            + [f'mTAP@{k}' for k in tols] + ['tIoUF1'])
    draws = {k: [] for k in keys}
    for _ in range(n_draw):
        rg = {}
        for c in CLASSES:
            n = len(gt_ev[c])
            rg[c] = [dict(gt_ev[c][rng.randrange(n)]) for _ in range(n)] if n else []
        lg, mm, ng, npd = event_metrics(rg, pr_ev, tols, dist_fn)
        for k in tols:
            draws[f'F1@{k}'].append(mm[k][0])
            draws[f'mF1@{k}'].append(mm[k][1])
            draws[f'mTAP@{k}'].append(mm[k][2])
        ltp = sum(lg[c][0] for c in CLASSES)
        draws['tIoUF1'].append(prf(ltp, ng, npd)[2])
    return {k: (sorted(v)[int(0.025 * n_draw)], sorted(v)[int(0.975 * n_draw)])
            for k, v in draws.items()}


def bootstrap_frame(pairs_by_track, n_draw, rng):
    """按 (vid,gtid) 轨迹重采样 -> 帧级 macroF1 95% CI"""
    keys = list(pairs_by_track)
    if not keys:
        return (0.0, 0.0)
    vals = []
    for _ in range(n_draw):
        pairs = []
        for _ in range(len(keys)):
            pairs.extend(pairs_by_track[rng.choice(keys)])
        vals.append(cm_metrics([(g, p) for g, p, *_ in pairs])['macroF1'])
    vals.sort()
    return vals[int(0.025 * n_draw)], vals[int(0.975 * n_draw)]


def evaluate_tag(tag, videos, args):
    """-> 汇总 dict (多视频池化) + 逐视频明细打印"""
    all_pairs = []                       # (gt, pred, vid, gtid, frame)
    pairs_by_track = defaultdict(list)   # (vid,gtid) -> [(gt,pred,vid,gtid,frame)]
    gt_events = {c: [] for c in CLASSES}
    pr_events = {c: [] for c in CLASSES}
    n_gt_frames = 0
    per_vid_frame = {}
    anchor_of = (lambda e: (e['s'] + e['e']) // 2) if args.anchor == 'center' \
        else (lambda e: e['s'])
    dist_fn = make_dist_fn(args.mode, anchor_of)

    for vid in videos:
        gt = load_gt(vid, args.xml_dir)
        pb, tl = load_pred(vid, tag, args.runs_root)
        mapping = match_tracks(pb, gt, args.iou)
        v_n_gt = sum(len(d) for d in gt.values())
        n_gt_frames += v_n_gt
        v_pairs = []
        mapped_gtids = set()
        for tid, gtid in sorted(mapping.items()):
            d = gt.get(gtid)
            if d is None or tid not in tl or not tl[tid]:
                continue
            mapped_gtids.add(gtid)
            preds = tl[tid]
            g_frames = sorted(d)
            lab_fill, prob_fill = forward_fill(preds)
            for gf in g_frames:
                if gf < preds[0][0] or gf > preds[-1][0]:
                    continue
                item = (d[gf][1], lab_fill[gf], vid, gtid, gf)
                all_pairs.append(item)
                v_pairs.append(item)
                pairs_by_track[(vid, gtid)].append(item)
            # 预测事件(映射轨迹)
            for s, e, lab in build_segments(lab_fill):
                pr_events[lab].append(dict(vid=vid, gtid=gtid, s=s, e=e,
                                           score=seg_score((s, e, lab), prob_fill)))
        per_vid_frame[vid] = cm_metrics([(g, p) for g, p, *_ in sorted(
            v_pairs, key=lambda x: (x[2], x[3], x[4]))]) | {'e2e': sum(
                1 for g, p, *_ in v_pairs if g == p) / max(v_n_gt, 1)}
        # GT 事件(每映射轨迹只登记一次, 防多 pred 轨迹映射同一 GT 时双计数)
        for gtid in mapped_gtids:
            d = gt[gtid]
            for s, e, lab in build_segments({gf: d[gf][1] for gf in sorted(d)}):
                gt_events[lab].append(dict(vid=vid, gtid=gtid, s=s, e=e))

    # ---- 帧级 ----
    ordered = sorted(all_pairs, key=lambda x: (x[2], x[3], x[4]))
    pairs_gp = [(g, p) for g, p, *_ in ordered]
    fr = cm_metrics(pairs_gp)
    fr_bal = cm_metrics([(g, p) for g, p, *_ in truncated_pairs(ordered)])
    e2e = sum(1 for g, p in pairs_gp if g == p) / max(n_gt_frames, 1)
    rng = random.Random(0)
    ci_f1 = bootstrap_frame(pairs_by_track, args.bootstrap, rng) if args.bootstrap else None

    # ---- 事件级 ----
    lg, mm, ng, npd = event_metrics(gt_events, pr_events, args.tols, dist_fn)
    ltp = sum(lg[c][0] for c in CLASSES)
    _, _, tiou_f1 = prf(ltp, ng, npd)
    k0, k1 = args.tols[0], args.tols[-1]
    mtap = {k: mm[k][2] for k in args.tols}
    f1k = {k: mm[k][0] for k in args.tols}
    mf1k = {k: mm[k][1] for k in args.tols}
    gain = mtap[k1] - mtap[k0]              # 容差增益: >0 越大=时间定位越量化
    ci_ev = bootstrap_event(gt_events, pr_events, args.tols, dist_fn,
                            args.bootstrap, rng) if args.bootstrap else None

    # ---- 打印 ----
    print(f'\n===== {tag} =====')
    for vid, m in per_vid_frame.items():
        print(f'  视频{vid}: macroF1={m["macroF1"]:.3f} micro={m["microF1"]:.3f} '
              f'macroRec={m["macroRec"]:.3f} 端到端={m["e2e"]:.3f}')
    print(f'GT帧·人={n_gt_frames} 覆盖对={len(pairs_gp)} | GT事件={ng} 预测事件={npd}')
    print(f'{"class":9s} {"GT/预":>8s} {"tIoU TP":>7s} {"tIoUF1":>7s} '
          + ' '.join(f'{"F1@" + str(k):>7s}' for k in args.tols))
    for c in CLASSES:
        row = ' '.join(f'{prf(*mm[k][3][c][:3])[2]:7.3f}' for k in args.tols)
        print(f'{c:9s} {lg[c][1]:4d}/{lg[c][2]:<3d} {lg[c][0]:7d} '
              f'{prf(*lg[c][:3])[2]:7.3f} {row}')
    bal = f' (截断平衡: macroF1={fr_bal["macroF1"]:.3f} micro={fr_bal["microF1"]:.3f})'
    print(f'帧级: macroF1={fr["macroF1"]:.3f} micro={fr["microF1"]:.3f} '
          f'macroRec={fr["macroRec"]:.3f} 端到端={e2e:.3f}{bal}')
    print(f'事件级: tIoU0.3 F1={tiou_f1:.3f} | '
          + ' '.join(f'mF1@{k}={mf1k[k]:.3f}' for k in args.tols) + ' | '
          + ' '.join(f'mTAP@{k}={mtap[k]:.3f}' for k in args.tols)
          + f' | 容差增益({k1}-{k0})={gain:+.3f}')
    if ci_ev:
        print('CI95(事件重采样): ' + '  '.join(
            f'{k}={v[0]:.3f}~{v[1]:.3f}' for k, v in ci_ev.items()))
    if ci_f1:
        print(f'CI95(轨迹重采样): macroF1={ci_f1[0]:.3f}~{ci_f1[1]:.3f}')

    out = dict(tag=tag, **{f'frame_{k}': v for k, v in fr.items()},
               frame_macroF1_ci=ci_f1, frame_bal_macroF1=fr_bal['macroF1'],
               frame_bal_micro=fr_bal['microF1'], e2e=e2e, tIoUF1=tiou_f1,
               tIoUF1_ci=ci_ev['tIoUF1'] if ci_ev else None,
               n_gt_ev=ng, n_pr_ev=npd, gain=gain)
    for k in args.tols:
        out[f'F1@{k}'] = f1k[k]
        out[f'mF1@{k}'] = mf1k[k]
        out[f'mTAP@{k}'] = mtap[k]
    if ci_ev:
        out['mF1at0_ci'] = ci_ev[f'mF1@{k0}']
        out['mF1atMax_ci'] = ci_ev[f'mF1@{k1}']
        out['mTAP0_ci'] = ci_ev[f'mTAP@{k0}']
        out['mTAPMax_ci'] = ci_ev[f'mTAP@{k1}']
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tags', nargs='+', required=True)
    ap.add_argument('--videos', nargs='+', default=['3', '5'])
    ap.add_argument('--iou', type=float, default=0.3)
    ap.add_argument('--bootstrap', type=int, default=1000)
    ap.add_argument('--mode', choices=['interval', 'anchor'], default='interval')
    ap.add_argument('--anchor', choices=['onset', 'center'], default='onset')
    ap.add_argument('--tols', nargs='+', type=int, default=[0, 2, 5, 10],
                    help='容差帧数@60fps, 默认 0/2/5/10 ≈ VNL {0,1,2,4}@25fps')
    ap.add_argument('--xml-dir', default=XML_DIR,
                    help='CVAT XML GT 目录(默认 2版; hold-out 传独立目录)')
    ap.add_argument('--runs-root', dest='runs_root', default=RUNS_10,
                    help='deploy_e2e_<tag> 所在根目录(默认 1.0 runs; 离线解码输出传 2.0 runs)')
    ap.add_argument('--label', default='',
                    help='汇总文件名附加标识, 区分不同 runs-root 的结果')
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    results = []
    for tag in args.tags:
        try:
            results.append(evaluate_tag(tag, args.videos, args))
        except FileNotFoundError as ex:
            print(f'\n===== {tag} 跳过: {ex.filename} 不存在 =====')

    if not results:
        return
    print(f'\n\n########## 汇总 (mode={args.mode}, anchor={args.anchor}, '
          f'tols={args.tols}@60fps, videos={args.videos}, bootstrap={args.bootstrap}) '
          f'##########')
    k0, k1 = args.tols[0], args.tols[-1]
    hdr = (f'{"tag":18s} {"macroF1":>16s} {"balF1":>6s} {"端到端":>6s} {"tIoUF1":>16s} '
           f'{f"mF1@{k0}":>16s} {f"mF1@{k1}":>16s} {f"mTAP@{k0}":>16s} '
           f'{f"mTAP@{k1}":>16s} {"容差增益":>7s}')
    print(hdr)
    for r in results:
        def cell(v, ci):
            return f'{v:.3f}[{ci[0]:.2f},{ci[1]:.2f}]' if ci else f'{v:.3f}'
        print(f'{r["tag"]:18s} {cell(r["frame_macroF1"], r["frame_macroF1_ci"]):>16s} '
              f'{r["frame_bal_macroF1"]:6.3f} {r["e2e"]:6.3f} '
              f'{cell(r["tIoUF1"], r["tIoUF1_ci"]):>16s} '
              f'{cell(r[f"mF1@{k0}"], r.get("mF1at0_ci")):>16s} '
              f'{cell(r[f"mF1@{k1}"], r.get("mF1atMax_ci")):>16s} '
              f'{cell(r[f"mTAP@{k0}"], r.get("mTAP0_ci")):>16s} '
              f'{cell(r[f"mTAP@{k1}"], r.get("mTAPMax_ci")):>16s} {r["gain"]:+7.3f}')

    cols = ['tag', 'frame_macroF1', 'frame_bal_macroF1', 'frame_bal_micro', 'e2e',
            'tIoUF1', 'gain', 'n_gt_ev', 'n_pr_ev']
    cols += ([f'F1@{k}' for k in args.tols] + [f'mF1@{k}' for k in args.tols]
             + [f'mTAP@{k}' for k in args.tols])
    suffix = f'{args.mode}_{args.anchor}_' + '_'.join(map(str, args.tols))
    if args.label:
        suffix += f'_{args.label}'
    csv_p = os.path.join(OUT_DIR, f'summary_{suffix}.csv')
    with open(csv_p, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in results:
            w.writerow([r[c] for c in cols])
    print(f'\n汇总 CSV: {csv_p}')


if __name__ == '__main__':
    main()
