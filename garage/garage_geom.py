"""真实障碍模型：墙 + 停靠车辆 + 分隔带（不是"y>11 全封"）"""
from __future__ import annotations
import numpy as np
import garage_env as G

WALL = [(0.0, 0.0, 30.0, 0.25), (0.0, G.SLOT_BACK, 30.0, G.SLOT_BACK + 0.3)]
SIDE = [(0.0, 0.0, 0.25, G.SLOT_BACK), (29.75, 0.0, 30.0, G.SLOT_BACK)]


def parked_cars():
    """非目标车位里的车（矩形 x1,y1,x2,y2）"""
    out = []
    for i in range(G.N_SLOT):
        if i + 1 == G.TARGET:
            continue
        x0 = G.SLOT_X0 + i * (G.SLOT_W + G.SLOT_DIV)
        y1 = G.AISLE_Y + 0.15
        out.append((x0 + 0.5, y1, x0 + 0.5 + 1.80, y1 + 4.40))
    return out


def dividers():
    out = []
    for i in range(G.N_SLOT + 1):
        x = G.SLOT_X0 + i * (G.SLOT_W + G.SLOT_DIV) - G.SLOT_DIV
        out.append((x, G.AISLE_Y + 0.15, x + G.SLOT_DIV, G.SLOT_BACK))
    return out


OBSTACLES = parked_cars() + dividers() + WALL + SIDE


def car_corners(S):
    x, y, th = S[0], S[1], S[2]
    fx, fy = np.cos(th), np.sin(th)
    lx, ly = -fy, fx
    return [(x + dx * fx + dy * lx, y + dx * fy + dy * ly)
            for dx, dy in ((G.FRONT_OH, G.WID / 2), (G.FRONT_OH, -G.WID / 2),
                           (-G.REAR_OH, -G.WID / 2), (-G.REAR_OH, G.WID / 2))]


def poly_hits_rect(pts, rect, margin=0.0):
    """凸多边形与轴对齐矩形是否相交（分离轴定理）"""
    x1, y1, x2, y2 = rect
    x1 -= margin; y1 -= margin; x2 += margin; y2 += margin
    rp = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    for poly, other in ((pts, rp), (rp, pts)):
        for i in range(len(poly)):
            ax, ay = poly[i]
            bx, by = poly[(i + 1) % len(poly)]
            nx, ny = -(by - ay), (bx - ax)
            pa = [nx * p[0] + ny * p[1] for p in pts]
            pb = [nx * p[0] + ny * p[1] for p in rp]
            if max(pa) < min(pb) or max(pb) < min(pa):
                return False
    return True


def free_pose(S, margin=0.06):
    pts = car_corners(S)
    for px, py in pts:
        if px < 0.2 or px > 29.8 or py < 0.2 or py > G.SLOT_BACK:
            return False
    for rect in OBSTACLES:
        if poly_hits_rect(pts, rect, margin):
            return False
    return True
