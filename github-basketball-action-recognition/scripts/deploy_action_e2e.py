# -*- coding: utf-8 -*-
"""端到端部署验证：YOLO26s检测(ByteTrack) -> 每ID 16帧clip缓冲(30Hz) -> R(2+1)D分类
-> 状态机约束的在线 Viterbi 解码 -> 标注视频 + 时间线CSV + 分段表 + FPS。
用法: python deploy_action_e2e.py --video 3
注意: VideoWriter 不支持中文路径，标注视频先写 ASCII 目录 outputs，结束自动搬回 runs。"""
import os
import math
import argparse
import csv
import shutil
import time
from collections import deque, defaultdict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
from ultralytics import YOLO

DET_W = r'F:\zjl\basket\runs\detect_26s_pball_os2x_cw112_val35\weights\best.pt'
BALL_W = r'F:\zjl\basket\runs\detect_26s_ballonly_os_val35\weights\best.pt'  # 纯球专项(双球源)
CLS_CK = r'F:\zjl\basket\二阶段起步\runs\clip_R21D_action\val35_sliding\best.pt'
RUNS = r'F:\zjl\basket\二阶段2.0\runs'   # 2.0 输出目录(权重等输入路径不变)
TMP_OUT = r'F:\zjl\basket\二阶段起步\outputs'   # ASCII 路径，VideoWriter 需要
VIDDIR = r'F:\zjl\basket\对应视频2'
CLASSES = ['normal', 'dribble', 'defense', 'shoot', 'pass']
COLORS = {c: tuple(int(x) for x in v) for c, v in zip(
    CLASSES, [(96, 96, 96), (0, 165, 255), (255, 0, 255), (0, 0, 255), (0, 200, 0)])}

CLIP_N = 16        # clip 帧数（30Hz）
SAMPLE_EVERY = 2   # 源视频 60fps -> 每2帧取1帧 = 30Hz（与训练一致）
PRED_EVERY = 8     # 缓冲每新增8帧推理一次（每16个源帧 ~0.27s）
CTX = 0.20         # 与训练一致的框外扩
MIN_H = 24         # 忽略过小人框
CONF = 0.35

# 状态机转移矩阵（用户定义的语法 + analyze_action_transitions.py 的经验统计）
# 行=当前状态, 列=下一状态；对角=自环（同一动作跨多个观测步持续）。
# dribble->normal=0.01 是语法外的低概率"逃生口"（应对漏检/失误，非主路径）。
TRANS = {
    'normal':  {'normal': 0.750, 'dribble': 0.115, 'defense': 0.130, 'shoot': 0.0025, 'pass': 0.0025},
    'dribble': {'normal': 0.010, 'dribble': 0.800, 'defense': 0.0025, 'shoot': 0.085, 'pass': 0.105},
    'defense': {'normal': 0.190, 'dribble': 0.007, 'defense': 0.800, 'shoot': 0.0015, 'pass': 0.0015},
    'shoot':   {'normal': 0.500, 'dribble': 0.0025, 'defense': 0.001, 'shoot': 0.495, 'pass': 0.0015},
    'pass':    {'normal': 0.500, 'dribble': 0.0025, 'defense': 0.001, 'shoot': 0.0015, 'pass': 0.495},
}


class ViterbiSmoother:
    """在线 Viterbi 解码：把逐步 softmax 观测解码为满足状态机语法的状态序列。
    每个 track 独立维护 delta(对数路径概率) 与 psi(回溯指针)，每步取当前最优状态输出。
    相比软投票：既不稀释短轨迹置信度，又通过转移概率抑制跳变（如 normal<->defense 抖动）。"""

    def __init__(self, classes, trans, floor=1e-6):
        self.classes = classes
        self.idx = {c: i for i, c in enumerate(classes)}
        self.floor = floor
        self.logA = {s: [math.log(max(trans[s][t], floor)) for t in classes] for s in classes}
        self.delta = defaultdict(list)   # tid -> [[logp per state], ...]
        self.psi = defaultdict(list)     # tid -> [[回溯指针 per state], ...]

    def update(self, tid, probs):
        """probs: dict/list，类目概率（softmax 输出）。返回 (state, posterior_prob)。"""
        logB = [math.log(max(probs[self.idx[c]], self.floor)) for c in self.classes]
        dl = self.delta[tid]
        if not dl:
            cur = logB[:]
            self.psi[tid].append([-1] * len(self.classes))
        else:
            prev = dl[-1]
            cur, bk = [], []
            for j in range(len(self.classes)):
                best, arg = -1e18, 0
                for i in range(len(self.classes)):
                    v = prev[i] + self.logA[self.classes[i]][j]
                    if v > best:
                        best, arg = v, i
                cur.append(best + logB[j])
                bk.append(arg)
            self.psi[tid].append(bk)
        dl.append(cur)
        m = max(cur)
        e = [math.exp(v - m) for v in cur]
        s = sum(e)
        j = int(np.argmax(e))
        return self.classes[j], e[j] / s

    def force_state(self, tid, cls_name):
        """物理事件硬写状态(球离体=shoot/pass 实锤): 不经转移矩阵,
        该步 delta 置为事件类独热(路径概率继承前序最优), 下一步恢复语法转移。"""
        dl = self.delta[tid]
        j = self.idx[cls_name]
        n = len(self.classes)
        if not dl:
            cur = [-1e18] * n
            cur[j] = 0.0
            self.psi[tid].append([-1] * n)
        else:
            prev = dl[-1]
            i = int(np.argmax(prev))
            cur = [-1e18] * n
            cur[j] = prev[i]
            self.psi[tid].append([i] * n)
        dl.append(cur)


CLS_SPEC = {'arch': 'r2plus1d18', 'size': 112,
            'mean': (0.43216, 0.394666, 0.37645),
            'std': (0.22803, 0.22145, 0.216989)}


def build_model(dev, ck_path=None):
    """按 checkpoint 元信息构建分类器: arch=r2plus1d18(默认)/mvitv2s(224 输入)"""
    ck = torch.load(ck_path or CLS_CK, map_location=dev)
    arch = ck.get('arch', 'r2plus1d18')
    if arch == 'mvitv2s':
        m = torchvision.models.video.mvit_v2_s(weights=None)
        hin = m.head[-1].in_features if isinstance(m.head, (nn.Sequential,)) \
            else m.head.in_features
        m.head = nn.Linear(hin, len(CLASSES))
        CLS_SPEC.update(arch=arch, size=224,
                        mean=(0.45, 0.45, 0.45), std=(0.225, 0.225, 0.225))
    else:
        m = torchvision.models.video.r2plus1d_18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, len(CLASSES))
        CLS_SPEC.update(arch=arch, size=112,
                        mean=(0.43216, 0.394666, 0.37645),
                        std=(0.22803, 0.22145, 0.216989))
    m.load_state_dict(ck['model'])
    print(f'分类器: {arch} 输入={CLS_SPEC["size"]} ckpt_macroF1={ck.get("val_macroF1")}')
    return m.to(dev).eval()


@torch.no_grad()
def classify(model, dev, batch):
    """batch: list of (16,112,112,3) uint8 -> list of (label, prob, prob_vec)。
    MViT: 112 图块按 checkpoint 尺寸重采样(224) + 对应归一化; 224 时分块前向防 OOM"""
    size = CLS_SPEC['size']
    arr = np.stack(batch)
    if size != arr.shape[2]:
        arr = np.stack([cv2.resize(f, (size, size))
                        for clip in arr for f in clip]).reshape(
                            len(batch), 16, size, size, 3)
    x = torch.from_numpy(arr).float().permute(0, 4, 1, 2, 3).to(dev) / 255.0
    mean = torch.tensor(CLS_SPEC['mean'], device=dev).view(1, 3, 1, 1, 1)
    std = torch.tensor(CLS_SPEC['std'], device=dev).view(1, 3, 1, 1, 1)
    x = (x - mean) / std
    step = 8 if size == 224 else max(len(x), 1)
    p = torch.cat([torch.softmax(model(x[i:i + step]), dim=1)
                   for i in range(0, len(x), step)])
    out = []
    for row in p:
        i = int(row.argmax())
        out.append((CLASSES[i], float(row[i]), [float(v) for v in row]))
    return out


class BallMonitor:
    """固定机位球运动学(用户核心思想: 球附近的人=持球人, 持球->dribble,
    球高速离体->shoot(升)/pass(横))。机位固定 => 像素速度≈物理速度, 无需相机补偿。
    持球判定: 球心落在某检测框内(外扩10%), 取框最小者(离球最近的人);
    离体事件: 距上次持球<=6帧且球速>vmin(px/帧), 按方向二分;
    加成: 事件后 hold 帧内对该 track 的对应类概率加 b, 归一化后交给解码器
    (Viterbi 语法仍生效: 持球期 dribble 加成先建立状态, 离体事件即可合法转入 shoot/pass)。"""

    def __init__(self, classes, vmin=15.0, hold=30, boost=0.35, own_boost=0.40,
                 do_poss=True, do_detach=True, do_pass2=True, fill=0,
                 do_def=False, def_prox=1.4, def_persist=45):
        self.idx = {c: i for i, c in enumerate(classes)}
        self.do_poss, self.do_detach, self.do_pass2 = do_poss, do_detach, do_pass2
        self.fill = fill
        self.n_fill = 0
        self.do_def, self.def_prox, self.def_persist = do_def, def_prox, def_persist
        self.persist = {}                   # (tid, owner_tid) -> 持续贴近帧数
        self.n_def = 0
        self.n_def_skip = 0                 # 因同队/未知队被门控挡掉的贴近帧
        self.team = {}                      # tid -> 0/1/-1, 由 TeamEstimator 每帧回填
        self.vmin, self.hold, self.b_ev, self.b_own = vmin, hold, boost, own_boost
        self.last = None                    # (fi, cx, cy)
        self.owner = None                   # (tid, fi) 最近持球人
        self.lost = None                    # (fi, tid) 持球后失球(pass两点启发式)
        self.events = defaultdict(deque)    # tid -> deque[(until_fi, cls_idx, b)]
        self.forces = defaultdict(deque)    # tid -> deque[(until_fi, cls_idx)] 硬写队列
        self.n_shoot = self.n_pass = 0
        self.n_poss = 0
        self.n_nodef = 0
        self.n_nodrib = 0

    def step(self, fi, balls, persons):
        """每源帧调用。balls=[(cx,cy)], persons={tid:(x1,y1,x2,y2)}"""
        if not balls:
            if self.fill and self.last is not None and fi - self.last[0] <= self.fill:
                self.n_fill += 1                    # 短断档: 球位保持, 持球判定照常(不置 lost)
                self._poss_at(fi, self.last[1], self.last[2], persons)
            else:
                if self.last is not None and self.owner and fi - self.owner[1] <= 6:
                    self.lost = (fi, self.owner[0])  # 持球人刚丢球(检测断档起点)
                self.last = None  # 球丢失即重置, 重现后重新起算速度
            return
        if self.last and fi - self.last[0] <= max(4, self.fill):
            _, px, py = self.last
            cx, cy = min(balls, key=lambda p: (p[0] - px) ** 2 + (p[1] - py) ** 2)
            sp = (((cx - px) ** 2 + (cy - py) ** 2) ** 0.5) / (fi - self.last[0])
            if self.do_detach and sp > self.vmin and self.owner and fi - self.owner[1] <= 6:
                dx, dy = cx - px, cy - py
                if dy < -0.5 * sp:                      # 上升主导 -> 投篮
                    self._fire(fi, self.owner[0], self.idx['shoot'], self.b_ev, force=True)
                    self.n_shoot += 1
                elif abs(dx) > abs(dy):                 # 横向主导 -> 传球
                    self._fire(fi, self.owner[0], self.idx['pass'], self.b_ev, force=True)
                    self.n_pass += 1
            self.last = (fi, cx, cy)
        else:
            self.last = (fi, balls[0][0], balls[0][1])
        # pass 两点启发式: 飞行段球不可见导致速度断档, 改用"失球→他人框内重现"判传球
        if self.do_pass2 and self.lost is not None:
            lfi, ltid = self.lost
            cx, cy = self.last[1], self.last[2]
            if fi - lfi <= 30 and any(
                    tid != ltid and x1 <= cx <= x2 and y1 <= cy <= y2
                    for tid, (x1, y1, x2, y2) in persons.items()):
                self._fire(fi, ltid, self.idx['pass'], self.b_ev, force=True)
                self.n_pass += 1
            self.lost = None    # 重现即消费(成功与否)
        # 持球判定(当帧): 球心落在某人人框内, 取最小框
        self._poss_at(fi, self.last[1], self.last[2], persons)
        # 对位防守启发式: 非持球人中心持续贴近持球人(人盯人签名) -> defense 加成
        if self.do_def and self.owner is not None:
            otid = self.owner[0]
            ob = persons.get(otid)
            if ob is not None:
                ox, oy = (ob[0] + ob[2]) / 2, (ob[1] + ob[3]) / 2
                oh = ob[3] - ob[1]
                for tid, (x1, y1, x2, y2) in persons.items():
                    if tid == otid:
                        continue
                    if self.team:                       # 异队门控: 防守持球人的大概率是另一队
                        ta, tb = self.team.get(tid, -1), self.team.get(otid, -1)
                        if ta < 0 or tb < 0 or ta == tb:
                            self.n_def_skip += 1
                            continue
                    d2 = ((x1 + x2) / 2 - ox) ** 2 + ((y1 + y2) / 2 - oy) ** 2
                    key = (tid, otid)
                    if d2 < (self.def_prox * oh) ** 2:
                        c = self.persist.get(key, (0,))[0] + 1
                        if c == self.def_persist:
                            self._fire(fi, tid, self.idx['defense'], self.b_ev)
                            self.n_def += 1
                            c = 0
                        self.persist[key] = (c,)
                    else:
                        self.persist.pop(key, None)

    def _poss_at(self, fi, cx, cy, persons):
        """持球判定: 球心(可为断档保持位)落在某人人框内(外扩10%), 取最小框(离球最近的人)"""
        best = None
        for tid, (x1, y1, x2, y2) in persons.items():
            mx, my = (x2 - x1) * 0.10, (y2 - y1) * 0.10
            if x1 - mx <= cx <= x2 + mx and y1 - my <= cy <= y2 + my:
                area = (x2 - x1) * (y2 - y1)
                if best is None or area < best[1]:
                    best = (tid, area)
        if self.do_poss and best is not None:
            self.owner = (best[0], fi)
            self.n_poss += 1
            self._fire(fi, best[0], self.idx['dribble'], self.b_own, hold=24)

    def _fire(self, fi, tid, idx, b, hold=None, force=False):
        q = self.events[tid]
        for it in list(q):
            if it[1] == idx and it[0] >= fi:    # 同类未过期, 无需重复
                return
        q.append((fi + (self.hold if hold is None else hold), idx, b))
        if force:
            self.forces[tid].append((fi + self.hold, idx))

    def take_force(self, tid, fi):
        """取该 track 当前未过期的硬写事件(有则消费并返回类索引)"""
        q = self.forces.get(tid)
        if not q:
            return None
        while q and q[0][0] < fi:
            q.popleft()
        return q.popleft()[1] if q else None

    def boost(self, tid, fi, pvec):
        """返回加成并归一化后的概率向量(无未过期事件则原样返回)"""
        q = self.events.get(tid)
        if not q:
            return pvec
        while q and q[0][0] < fi:
            q.popleft()
        if not q:
            return pvec
        v = list(pvec)
        for _, idx, b in q:
            v[idx] += b
        s = sum(v)
        return [x / s for x in v]

    def semantic_gate(self, tid, fi, pvec, nodef, nodrib):
        """妨碍语义门控(2026-10-28 用户澄清: defense=妨碍者行为, 持球人自身不会 defense;
        dribble=持球人专属动作)。持球判定新鲜(<=6帧)时软抑制, 仅压低不改排他:
        持球人 defense 概率 x nodef, 非持球人 dribble 概率 x nodrib(0=关)。"""
        if not self.owner or fi - self.owner[1] > 6:
            return pvec
        otid = self.owner[0]
        v = list(pvec)
        hit = False
        if nodef > 0 and tid == otid and v[self.idx['defense']] > 0:
            v[self.idx['defense']] *= nodef
            self.n_nodef += 1
            hit = True
        if nodrib > 0 and tid != otid and v[self.idx['dribble']] > 0:
            v[self.idx['dribble']] *= nodrib
            self.n_nodrib += 1
            hit = True
        if not hit:
            return pvec
        s = sum(v)
        return [x / s for x in v] if s > 0 else v


class TeamEstimator:
    """队伍归属(外形特征): 按 track ID 累积躯干 Lab 色度(a,b) 长期均值, 每 every 源帧
    对可信 ID(帧数>=min_frames)做 k-means++ 多次重启(k=2, 带最小簇规模约束) -> 两队标签。
    用 ID 稳定性沉淀长期外观, 比逐帧判色抗遮挡/姿态/光照抖动。
    仅服务对位防守门控: 能防持球人的大概率是另一队, 同队/未知队不加成。
    距离比不准的 ID 记 -1(未知, 裁判/观众/队服撞色) -> 不参与加成。
    诊断依据 diag_team_color.py: 两视频 Lab 色度队间/队内均距比 2.08/2.63 (>1.5 可分)。"""

    def __init__(self, every=60, min_frames=30, margin=0.80, restarts=12):
        self.lab = {}        # tid -> (sum_a, sum_b) 躯干高饱和像素 Lab 色度累加
        self.cnt = {}        # tid -> 累积有效帧数
        self.every, self.min_frames, self.margin = every, min_frames, margin
        self.restarts = restarts
        self.teams = {}      # tid -> 0/1/-1
        self.last_k = 0
        self.n_k = 0
        self.n_fail = 0      # 聚类不可信(簇失衡/ID太少)的次数

    def update(self, fi, frame, persons):
        for tid, (x1, y1, x2, y2) in persons.items():
            w, h = x2 - x1, y2 - y1
            tx1, tx2 = int(x1 + 0.25 * w), int(x2 - 0.25 * w)
            ty1, ty2 = int(y1 + 0.20 * h), int(y1 + 0.50 * h)   # 躯干核心: 避开头/短裤/边缘背景
            tx1, ty1 = max(0, tx1), max(0, ty1)
            tx2, ty2 = min(frame.shape[1], tx2), min(frame.shape[0], ty2)
            if tx2 - tx1 < 6 or ty2 - ty1 < 8:
                continue
            crop = frame[ty1:ty2, tx1:tx2]
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            m = (hsv[..., 1] > 60) & (hsv[..., 2] > 40)
            if int(m.sum()) < 0.10 * m.size:        # 有效像素过少(白/灰/黑队服或背景占比高)
                continue
            lb = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)[m].mean(0)   # 只统计高饱和像素
            sa, sb = self.lab.get(tid, (0.0, 0.0))
            self.lab[tid] = (sa + lb[1] - 128, sb + lb[2] - 128)
            self.cnt[tid] = self.cnt.get(tid, 0) + 1

    def _kmeans2(self, X, seed):
        """k-means++ 初始化 + Lloyd 迭代(k=2), 固定种子保证可复现"""
        rng = np.random.default_rng(seed)
        n = len(X)
        C = [X[int(rng.integers(n))]]
        d2 = ((X - C[0]) ** 2).sum(1)
        C.append(X[int(rng.choice(n, p=d2 / max(d2.sum(), 1e-12)))] if d2.sum() > 0 else X[0])
        C = np.array(C, dtype=np.float64)
        a = None
        for _ in range(40):
            a = ((X[:, None, :] - C[None]) ** 2).sum(2).argmin(1)
            C2 = np.array([X[a == j].mean(0) if (a == j).sum() else C[j] for j in range(2)])
            if np.allclose(C2, C):
                break
            C = C2
        inertia = float(((X - C[a]) ** 2).sum())
        return inertia, a, C

    def recluster(self, fi):
        if fi - self.last_k < self.every:
            return
        self.last_k = fi
        ids = [t for t, c in self.cnt.items() if c >= self.min_frames]
        if len(ids) < 6:
            return
        X = np.array([[self.lab[t][0] / self.cnt[t], self.lab[t][1] / self.cnt[t]]
                      for t in ids], dtype=np.float64)
        best = None
        for s in range(self.restarts):
            inertia, a, C = self._kmeans2(X, seed=s)
            n0, n1 = int((a == 0).sum()), int((a == 1).sum())
            if min(n0, n1) < max(2, 0.20 * len(ids)):
                continue                                  # 拒掉"1个离群 vs 其余全部"式退化解
            if best is None or inertia < best[0]:
                best = (inertia, a.copy(), C.copy())
        if best is None:
            self.n_fail += 1
            self.teams = {}                               # 本视频不可分 -> 门控全关(安全兜底)
            return
        _, a, C = best
        d = np.sqrt(((X[:, None, :] - C[None, :, :]) ** 2).sum(2))
        teams = {}
        for i, t in enumerate(ids):
            s0, s1 = d[i, 0], d[i, 1]
            teams[t] = -1 if max(s0, s1) <= 1e-9 or min(s0, s1) >= self.margin * max(s0, s1) \
                else int(s0 > s1)
        self.teams = teams
        self.n_k += 1

    def summary(self):
        g = {0: [], 1: [], -1: []}
        for t, v in self.teams.items():
            g[v].append(t)
        return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', default='3', help='视频号(1-6)')
    ap.add_argument('--tag', default='sliding', help='输出目录后缀 deploy_e2e_<tag>')
    ap.add_argument('--ball-poss', action='store_true', dest='ball_poss',
                    help='球规则1: 持球判定(球心在谁框内取最小框)->dribble概率加成')
    ap.add_argument('--ball-detach', action='store_true', dest='ball_detach',
                    help='球规则2: 球高速离体方向二分 -> shoot(升)/pass(横) 硬写')
    ap.add_argument('--ball-pass2', action='store_true', dest='ball_pass2',
                    help='球规则3: 失球后30帧内他人框内重现 -> pass 硬写(两点启发式)')
    ap.add_argument('--ball-fill', type=int, default=0, dest='ball_fill',
                    help='球轨迹短程保持帧数: 断档<=N帧时用最后球位继续持球判定(0=关)')
    ap.add_argument('--def-prox', type=float, default=0.0, dest='def_prox',
                    help='对位防守规则: 非持球人中心距持球人<prox*持球人框高并持续 -> defense 加成(0=关)')
    ap.add_argument('--def-persist', type=int, default=45, dest='def_persist',
                    help='对位防守持续帧数阈值')
    ap.add_argument('--team-color', action='store_true', dest='team_color',
                    help='队伍归属门控: 按ID聚合躯干HSV直方图K-means分两队, 仅异队才触发对位防守加成')
    ap.add_argument('--team-margin', type=float, default=0.80, dest='team_margin',
                    help='队伍归属置信阈值: 最近/次近中心距离比<margin才算可信(越小越严格)')
    ap.add_argument('--ball-vmin', type=float, default=15.0, help='球速阈值 px/帧')
    ap.add_argument('--ball-boost', type=float, default=0.35, help='离体事件概率加成')
    ap.add_argument('--poss-nodef', type=float, default=0.0, dest='poss_nodef',
                    help='妨碍语义门控: 持球判定新鲜时持球人自身 defense 概率乘该系数(0=关, 建议 0.2)')
    ap.add_argument('--poss-nodrib', type=float, default=0.0, dest='poss_nodrib',
                    help='妨碍语义门控: 持球判定新鲜时非持球人 dribble 概率乘该系数(0=关, 建议 0.2)')
    ap.add_argument('--ball-hold', type=int, default=30, help='离体加成持续帧数')
    ap.add_argument('--dual-ball', action='store_true', dest='dual_ball',
                    help='并第二个纯球专项模型(几何去重取并集), 提升球覆盖率, 代价约一倍检测前向')
    ap.add_argument('--dump-boxes', action='store_true', dest='dump_boxes',
                    help='导出预测帧的轨迹框(boxes_N.csv), 供 GT 对齐评估')
    ap.add_argument('--pred-every', type=int, default=PRED_EVERY, dest='pred_every',
                    help='每新增N个采样帧(30Hz)推理一次, 越小时间分辨率越高、FPS越低')
    ap.add_argument('--decode', default='viterbi', choices=['viterbi', 'argmax'],
                    help='解码: viterbi=状态机约束在线解码(默认), argmax=每步直接取最大概率(无时序语法约束)')
    ap.add_argument('--smooth', type=int, default=0,
                    help='对每条轨迹最近N次预测做多数表决平滑(0=关), 抑制 argmax 逐帧抖动')
    ap.add_argument('--det-weights', dest='det_weights', default=DET_W,
                    help='一阶段检测器权重(默认 26s 人球联合)')
    ap.add_argument('--cls-weights', dest='cls_weights', default=CLS_CK,
                    help='二阶段动作分类器权重(默认 val35_sliding 基线)')
    ap.add_argument('--cls-weights2', dest='cls_weights2', default='',
                    help='第二分类器权重(32帧窗口); 提供则启用双窗口概率融合(16f+32f均值)')
    ap.add_argument('--dump-probs', action='store_true', dest='dump_probs',
                    help='落盘每次预测的解码前全类概率(probs_N.csv: frame,track_id,p_类...,force), 供离线解码实验')
    ap.add_argument('--no-video', action='store_true', dest='no_video',
                    help='不写标注视频(仅CSV, 提速省盘)')
    args = ap.parse_args()
    vid = args.video
    src = os.path.join(VIDDIR, f'{vid}.mp4')
    dev = 'cuda'
    os.makedirs(TMP_OUT, exist_ok=True)

    det = YOLO(args.det_weights)
    ball_det = YOLO(BALL_W) if args.dual_ball else None
    cls_model = build_model(dev, args.cls_weights)
    cls_model2 = build_model(dev, args.cls_weights2) if args.cls_weights2 else None
    buf_n = 32 if cls_model2 is not None else CLIP_N

    cap = cv2.VideoCapture(src)
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 60.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_path = os.path.join(TMP_OUT, f'annotated_{vid}.mp4')
    vw = None if args.no_video else cv2.VideoWriter(
        out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps_src, (W, H))

    bufs = defaultdict(lambda: deque(maxlen=buf_n))     # tid -> 112px RGB 帧队列(30Hz)
    ready = defaultdict(int)                            # tid -> 自上次推理新增采样帧数
    vit = ViterbiSmoother(CLASSES, TRANS)               # tid -> 状态机约束的在线解码
    cur = {}                                            # tid -> (当前解码状态, 后验概率)
    sm = defaultdict(lambda: deque(maxlen=max(args.smooth, 1)))   # 多数表决窗口
    mon = None
    if args.ball_poss or args.ball_detach or args.ball_pass2 or args.def_prox > 0:
        mon = BallMonitor(CLASSES, vmin=args.ball_vmin, hold=args.ball_hold,
                          boost=args.ball_boost, do_poss=args.ball_poss,
                          do_detach=args.ball_detach, do_pass2=args.ball_pass2,
                          fill=args.ball_fill, do_def=args.def_prox > 0,
                          def_prox=args.def_prox, def_persist=args.def_persist)
    team_est = TeamEstimator(margin=args.team_margin) if args.team_color else None
    timeline = []                                       # (frame, tid, label, prob)
    probsdump = []                                      # (frame, tid, *pvec, force) 解码前概率
    boxdump = []                                        # (frame, tid, x1, y1, x2, y2) 预测帧轨迹框
    fi = 0
    t_all = 0.0          # 帧间到达间隔 ≈ 每帧真实处理耗时（含检测+跟踪+绘制）
    t_cls = 0.0
    n_cls_calls = 0
    print(f'视频{vid} 球规则={"开" if mon else "关"} 模型=sliding: '
          f'{n_total}帧 @{fps_src:.0f}fps {W}x{H} -> {out_path}')

    stream = det.track(src, persist=True, stream=True, classes=[0, 1],
                       conf=CONF, imgsz=1280, tracker='bytetrack.yaml', verbose=False)
    prev = time.time()
    for r in stream:
        now = time.time()
        frame_ms = (now - prev) * 1000
        prev = now
        t_all += frame_ms
        fi += 1
        frame = r.orig_img
        sampled = (fi % SAMPLE_EVERY == 0)

        batch, keys, nfrs = [], [], []
        balls, persons = [], {}
        if r.boxes is not None and len(r.boxes):
            for b in r.boxes:
                if int(b.cls[0]) == 1:                  # 球: 只取中心点
                    bx1, by1, bx2, by2 = map(int, b.xyxy[0].tolist())
                    balls.append(((bx1 + bx2) / 2, (by1 + by2) / 2))
                    continue
                tid = int(b.id) if b.id is not None else -1
                if tid < 0:
                    continue
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                h = y2 - y1
                if h < MIN_H:
                    continue
                persons[tid] = (x1, y1, x2, y2)
                if sampled:  # 30Hz 采样入缓冲
                    cx1 = max(0, x1 - int((x2 - x1) * CTX))
                    cy1 = max(0, y1 - int(h * CTX))
                    cx2 = min(W, x2 + int((x2 - x1) * CTX))
                    cy2 = min(H, y2 + int(h * CTX))
                    patch = cv2.cvtColor(cv2.resize(frame[cy1:cy2, cx1:cx2], (112, 112)),
                                         cv2.COLOR_BGR2RGB)
                    bufs[tid].append(patch)
                    ready[tid] += 1
                if ready[tid] >= args.pred_every and len(bufs[tid]) >= CLIP_N \
                        and len(bufs[tid]) % CLIP_N == 0:
                    batch.append(np.stack(bufs[tid]))
                    keys.append(tid)
                    nfrs.append(len(bufs[tid]))
                    ready[tid] = 0

        if ball_det is not None:                        # 第二球源: 并集去重
            for b2 in ball_det.predict(frame, conf=CONF, imgsz=1280, verbose=False)[0].boxes:
                bx1, by1, bx2, by2 = map(int, b2.xyxy[0].tolist())
                cx, cy = (bx1 + bx2) / 2, (by1 + by2) / 2
                if all((cx - c[0]) ** 2 + (cy - c[1]) ** 2 > 576 for c in balls):
                    balls.append((cx, cy))

        if team_est is not None:
            team_est.update(fi, frame, persons)          # 按ID累积躯干色相(外形特征)
            team_est.recluster(fi)                       # 每60帧重聚两队
            if mon is not None:
                mon.team = team_est.teams                # 异队门控(先于球规则step)
        if mon is not None:
            mon.step(fi, balls, persons)                # 球运动学事件(每源帧)

        if batch:
            t1 = time.time()
            res = classify(cls_model, dev, [b[-CLIP_N:] for b in batch])
            if cls_model2 is not None:                  # 双窗口融合: 16f+32f 概率均值
                idx32 = [i for i, n in enumerate(nfrs) if n == buf_n]
                if idx32:
                    res32 = classify(cls_model2, dev, [batch[i] for i in idx32])
                    for i, (_l, _p, pv32) in zip(idx32, res32):
                        pv = res[i][2]
                        res[i] = (CLASSES[0], 0.0,
                                  [(a + b) / 2 for a, b in zip(pv, pv32)])
            t_cls += (time.time() - t1) * 1000
            n_cls_calls += 1
            for tid, (_lab, _pb, pvec) in zip(keys, res):
                if mon is not None:
                    pvec = mon.boost(tid, fi, pvec)     # 球事件概率加成(解码前)
                if mon is not None and (args.poss_nodef > 0 or args.poss_nodrib > 0):
                    pvec = mon.semantic_gate(tid, fi, pvec,
                                             args.poss_nodef, args.poss_nodrib)
                if args.decode == 'argmax':
                    j = int(max(range(len(pvec)), key=lambda k: pvec[k]))
                    lab_s, pb_s = CLASSES[j], pvec[j]
                else:
                    lab_s, pb_s = vit.update(tid, pvec)   # 状态机约束的在线 Viterbi 解码
                if args.dump_probs:
                    probsdump.append((fi, tid, *pvec, -1))
                if mon is not None:
                    fidx = mon.take_force(tid, fi)
                    if fidx is not None:              # 球离体=物理实锤, 硬写状态
                        if args.decode == 'viterbi':
                            vit.force_state(tid, CLASSES[fidx])
                        lab_s, pb_s = CLASSES[fidx], 0.90
                        if args.dump_probs:
                            probsdump.append((fi, tid, *pvec, fidx))
                        cur[tid] = (lab_s, pb_s)
                        timeline.append((fi, tid, lab_s, pb_s))
                        if args.dump_boxes and tid in persons:
                            boxdump.append((fi, tid, *persons[tid]))
                        continue
                if args.smooth > 0:                   # 短窗多数表决, 压 argmax 抖动
                    hist = sm[tid]
                    hist.append((lab_s, pb_s))
                    if len(hist) == args.smooth:
                        vote, psum = defaultdict(int), defaultdict(float)
                        for l_, p_ in hist:
                            vote[l_] += 1
                            psum[l_] += p_
                        m = max(vote.values())
                        lab_s = max([l_ for l_ in vote if vote[l_] == m], key=lambda l_: psum[l_])
                        pb_s = vote[lab_s] / len(hist)
                cur[tid] = (lab_s, pb_s)
                timeline.append((fi, tid, lab_s, pb_s))
                if args.dump_boxes and tid in persons:
                    boxdump.append((fi, tid, *persons[tid]))

        # 画框
        if mon is not None and mon.last and mon.last[0] == fi:
            cv2.circle(frame, (int(mon.last[1]), int(mon.last[2])), 4, (0, 255, 255), -1)
        if r.boxes is not None and len(r.boxes):
            for b in r.boxes:
                if int(b.cls[0]) == 1:
                    continue
                tid = int(b.id) if b.id is not None else -1
                if tid < 0:
                    continue
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                lab, pb = cur.get(tid, ('normal', 0.0))
                c = COLORS.get(lab, (255, 255, 255))
                cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
                cv2.putText(frame, f'ID{tid} {lab} {pb:.2f}', (x1, max(16, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2)
        cv2.putText(frame, f'f{fi} {frame_ms:.0f}ms', (12, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        if vw is not None:
            vw.write(frame)
        if fi % 300 == 0:
            print(f'  进度 {fi}/{n_total}  帧均值{t_all / fi:.0f}ms  分类调用{n_cls_calls}')

    cap.release()
    if vw is not None:
        vw.release()
    n = max(fi, 1)
    print(f'\n=== 运行完成 ===')
    print(f'帧数={fi}  管线帧均值={t_all / n:.1f}ms ({1000 / (t_all / n):.1f} FPS)')
    print(f'分类调用={n_cls_calls}次 累计={t_cls / 1000:.1f}s (均值{t_cls / max(n_cls_calls, 1):.0f}ms/批)')
    print(f'端到端吞吐={1000 / (t_all / n):.1f} FPS (含检测+跟踪+分类+绘制)')
    if mon is not None:
        print(f'球规则: 持球帧={mon.n_poss} 离体事件 shoot={mon.n_shoot} pass={mon.n_pass} '
              f'保持帧={mon.n_fill} 防守加成={mon.n_def} 语义门控 nodef={mon.n_nodef} nodrib={mon.n_nodrib} '
              f'同队/未知队挡掉={mon.n_def_skip}')
    if team_est is not None:
        g = team_est.summary()
        print(f'队伍归属: 重聚类{team_est.n_k}次 失败{team_est.n_fail}次 '
              f'队0={sorted(g[0])} 队1={sorted(g[1])} 未知={sorted(g[-1])}')

    # 时间线 CSV（unicode 安全）
    out_dir = os.path.join(RUNS, f'deploy_e2e_{args.tag}')
    os.makedirs(out_dir, exist_ok=True)
    csv_p = os.path.join(out_dir, f'timeline_{vid}.csv')
    with open(csv_p, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['frame', 'track_id', 'action_smooth', 'prob'])
        for row in timeline:
            w.writerow(row)
    print(f'时间线: {csv_p}  ({len(timeline)}条预测)')

    if args.dump_probs:
        pb_p = os.path.join(out_dir, f'probs_{vid}.csv')
        with open(pb_p, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['frame', 'track_id'] + [f'p_{c}' for c in CLASSES] + ['force'])
            w.writerows(probsdump)
        print(f'概率曲线: {pb_p}  ({len(probsdump)}条)')

    if args.dump_boxes:
        bx_p = os.path.join(out_dir, f'boxes_{vid}.csv')
        with open(bx_p, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['frame', 'track_id', 'x1', 'y1', 'x2', 'y2'])
            w.writerows(boxdump)
        print(f'轨迹框: {bx_p}  ({len(boxdump)}条)')

    # 分段表
    segs = []
    for tid in sorted({t for _, t, _, _ in timeline}):
        seq = [(fr, lab, pb) for fr, t, lab, pb in timeline if t == tid]
        start, cur = seq[0][0], seq[0][1]
        for i in range(1, len(seq) + 1):
            if i == len(seq) or seq[i][1] != cur:
                segs.append((tid, start, seq[i - 1][0], cur, seq[i - 1][2]))
                if i < len(seq):
                    start, cur = seq[i][0], seq[i][1]
    seg_p = os.path.join(out_dir, f'segments_{vid}.csv')
    with open(seg_p, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['track_id', 'start_f', 'end_f', 'action', 'prob'])
        for s in segs:
            w.writerow(s)
    print(f'分段表: {seg_p}  ({len(segs)}段)')
    for s in segs[:40]:
        print(f'  ID{s[0]} [{s[1]}-{s[2]}] {s[3]} ({s[4]:.2f})')

    if vw is not None:
        shutil.move(out_path, os.path.join(out_dir, f'annotated_{vid}.mp4'))
        print(f'标注视频已移至: {os.path.join(out_dir, f"annotated_{vid}.mp4")}')


if __name__ == '__main__':
    main()
