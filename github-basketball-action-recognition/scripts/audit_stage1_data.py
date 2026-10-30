# -*- coding: utf-8 -*-
"""阶段1 数据审计 (任务单 2_0_followup_20261030): 只读, 不修改任何标签。
输出:
  reports/2_0_followup_20261030_dataset_stats.json
  reports/2_0_followup_20261030_annotation_review_queue.csv  (疑似异常, 仅登记不改动)
审计内容:
  A. 视频元数据 (fps/帧数, cv2 实测)
  B. GT XML: 轨迹数/动作事件数(与评测器同规则 min_gap=6)/时长分布/边界越界/未知标签
  C. clip 数据集: 各划分各类 x 视频样本量 + 抽样完整性(16 jpg) + train/val 视频零重叠断言
"""
import os
import re
import json
import random
import csv
from collections import defaultdict

import cv2

import sys
sys.path.insert(0, r'F:\zjl\basket\二阶段起步\scripts')
from build_action_clip_dataset import parse_video, CLASSES  # noqa: E402

XML_DIR = r'F:\zjl\basket\2版'
VIDDIR = r'F:\zjl\basket\对应视频2'
DS = r'F:\zjl\basket\二阶段起步\datasets\dataset_action_clip_val35_sliding'
REPORTS = r'F:\zjl\basket\二阶段2.0\reports'
FPS_EXPECT = 60.0
MIN_GAP = 6           # 与 evaluate_v2.build_segments 相同
ANOM_SHORT = 10       # <10 帧 (<0.17s@60fps) 可疑
ANOM_LONG = 900       # >15s 可疑(1.0 场景动作极少超此)
SPLIT_VIDS = {'train': {'1', '2', '4', '6'}, 'val': {'3', '5'}}


def segments(frames_labels, min_gap=MIN_GAP):
    """与 evaluate_v2.build_segments 同规则: 连续同类段, 间隔<min_gap 合并"""
    segs, cur = [], None
    for fi in sorted(frames_labels):
        a = frames_labels[fi]
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


def audit_videos():
    out = {}
    for vid in '123456':
        p = os.path.join(VIDDIR, f'{vid}.mp4')
        cap = cv2.VideoCapture(p)
        fps = cap.get(cv2.CAP_PROP_FPS)
        nfr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        out[vid] = {'fps': round(fps, 3), 'frames': nfr,
                    'seconds': round(nfr / fps, 1) if fps else None,
                    'fps_ok': abs(fps - FPS_EXPECT) < 0.5,
                    'exists': os.path.exists(p)}
    return out


def audit_xml(vmeta):
    per_vid = {}
    queue = []
    for vid in '123456':
        xml = os.path.join(XML_DIR, f'{vid}.xml')
        n_tracks = 0
        n_action_frames = 0
        labels_seen = defaultdict(int)
        ev = defaultdict(list)          # cls -> [(gtid, s, e)]
        for ti, seq in parse_video(xml):
            n_tracks += 1
            fl = {}
            for fr, bb, act in seq:
                if act:
                    labels_seen[act] += 1
                    fl[fr] = act
            n_action_frames += len(fl)
            nfr_max = vmeta[vid]['frames']
            for s, e, lab in segments(fl):
                if lab not in CLASSES:
                    queue.append(dict(vid=vid, gtid=ti, cls=lab, start=s, end=e,
                                      dur=e - s + 1, reason='未知标签(不在5类映射)'))
                    continue
                ev[lab].append((ti, s, e))
                if e - s + 1 < ANOM_SHORT:
                    queue.append(dict(vid=vid, gtid=ti, cls=lab, start=s, end=e,
                                      dur=e - s + 1, reason=f'过短<{ANOM_SHORT}帧'))
                if e - s + 1 > ANOM_LONG:
                    queue.append(dict(vid=vid, gtid=ti, cls=lab, start=s, end=e,
                                      dur=e - s + 1, reason=f'过长>{ANOM_LONG}帧'))
                if nfr_max and (s < 0 or e >= nfr_max):
                    queue.append(dict(vid=vid, gtid=ti, cls=lab, start=s, end=e,
                                      dur=e - s + 1, reason='边界越界(超出视频帧数)'))
        dur = {c: sorted(e - s + 1 for _, s, e in ev[c]) for c in CLASSES}
        def stat(lst):
            if not lst:
                return dict(n=0)
            n = len(lst)
            return dict(n=n, mean=round(sum(lst) / n, 1),
                        median=lst[n // 2], min=lst[0], max=lst[-1],
                        mean_s=round(sum(lst) / n / FPS_EXPECT, 2))
        per_vid[vid] = {'tracks': n_tracks, 'action_frames': n_action_frames,
                        'events_per_class': {c: len(ev[c]) for c in CLASSES},
                        'label_frames': dict(labels_seen),
                        'duration_frames': {c: stat(dur[c]) for c in CLASSES}}
    return per_vid, queue


def audit_clips():
    out = {'per_class': {}, 'video_dist': {}, 'integrity_checked': 0,
           'integrity_failed': []}
    rng = random.Random(0)
    for split in ('train', 'val'):
        vids = defaultdict(lambda: defaultdict(int))
        for c in CLASSES:
            d = os.path.join(DS, split, c)
            names = sorted(os.listdir(d))
            out['per_class'][f'{split}/{c}'] = len(names)
            for name in names:
                vids[name.split('_')[0]][c] += 1
            for name in rng.sample(names, min(40, len(names))):
                cd = os.path.join(d, name)
                missing = [f'f{j:02d}.jpg' for j in range(16)
                           if not os.path.exists(os.path.join(cd, f'f{j:02d}.jpg'))]
                out['integrity_checked'] += 1
                if missing:
                    out['integrity_failed'].append(
                        dict(clip=f'{split}/{c}/{name}', missing=len(missing)))
        out['video_dist'][split] = {v: sum(m.values()) for v, m in sorted(vids.items())}
    tr = set(out['video_dist']['train'])
    va = set(out['video_dist']['val'])
    out['video_level_leakage'] = 'PASS' if not (tr & va) else f'FAIL: {tr & va}'
    return out


def main():
    vmeta = audit_videos()
    xml_stats, queue = audit_xml(vmeta)
    clip_stats = audit_clips()
    result = {'generated': '2026-10-30', 'fps_expect': FPS_EXPECT,
              'videos': vmeta, 'gt_xml': xml_stats, 'clips': clip_stats,
              'anomaly_count': len(queue),
              'anomaly_breakdown': dict(sorted(
                  {q['reason'].split('(')[0]: 1 for q in queue}.items()))}
    os.makedirs(REPORTS, exist_ok=True)
    with open(os.path.join(REPORTS, '2_0_followup_20261030_dataset_stats.json'),
              'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    with open(os.path.join(REPORTS, '2_0_followup_20261030_annotation_review_queue.csv'),
              'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['vid', 'gtid', 'cls', 'start', 'end', 'dur_frames',
                    'dur_seconds', 'reason'])
        for q in queue:
            w.writerow([q['vid'], q['gtid'], q['cls'], q['start'], q['end'],
                        q['dur'], round(q['dur'] / FPS_EXPECT, 2), q['reason']])
    # 控制台摘要
    print('== 视频元数据 ==')
    for v, m in vmeta.items():
        print(f"  v{v}: fps={m['fps']} frames={m['frames']} ({m['seconds']}s) fps_ok={m['fps_ok']}")
    print('== GT 事件 (min_gap=6 与评测器同规则) ==')
    tot = defaultdict(int)
    for v, m in xml_stats.items():
        line = ' '.join(f'{c}={m["events_per_class"][c]}' for c in CLASSES)
        print(f'  v{v}: tracks={m["tracks"]} act_frames={m["action_frames"]} | {line}')
        for c in CLASSES:
            tot[c] += m['events_per_class'][c]
    print('  全库合计: ' + ' '.join(f'{c}={tot[c]}' for c in CLASSES)
          + f'  总={sum(tot.values())}')
    print(f'== clips == 完整性抽检 {clip_stats["integrity_checked"]} 个, '
          f'失败 {len(clip_stats["integrity_failed"])}; '
          f'泄漏检查: {clip_stats["video_level_leakage"]}')
    print(f'== 异常清单 == {len(queue)} 条 -> annotation_review_queue.csv')


if __name__ == '__main__':
    main()
