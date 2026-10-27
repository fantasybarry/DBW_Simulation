"""Localization on the point cloud map (pointcloud_map.pcd, same frame as lanelet2_map.osm local_x/y).

Each Velodyne scan is ICP-matched to the map in 2D (x, y, yaw). Between scans (10 Hz) the pose is
predicted with the IMU gyro (yaw change) + GPS speed (FusedPose), so pure pursuit gets a fresh pose
at 50 Hz. Both clouds keep only "structure" points MIN_H..MAX_H above the ground (walls, poles,
trunks, curbs) projected flat; ground carries no x/y information.
"""
import math
import threading
import time

import numpy as np
from scipy.spatial import cKDTree

import velodyne

MIN_H, MAX_H = 0.5, 3.0          # structure band above ground (m)
GROUND_CELL_M = 10.0             # map ground = lowest point per cell this size (big: cells under trees
                                 # need some ground in them; slopes > ~5% would want smaller)
MAP_VOXEL_M  = 0.25
SCAN_VOXEL_M = 0.5
MIN_RANGE_M, MAX_RANGE_M = 3.0, 50.0   # closer = the car itself; farther = sparse + flat-ground error
INIT_DISTS   = (5.0, 3.0, 1.5, 0.8, 0.5)   # ICP match radii, coarse -> fine: first localization
TRACK_DISTS  = (1.5, 0.8, 0.4)             # ... per scan while tracking
ICP_ITERS    = 10
INIT_XY_M    = (-2.0, 0.0, 2.0)            # initial position guesses (grid) tried around the start guess...
INIT_YAWS_DEG = (0, 15, -15, 30, -30)      # ...times these headings
MIN_FIT      = 0.5               # inlier fraction (within the finest radius) to accept a match
MAX_JUMP_M   = 1.0               # tracking: reject a match this far from the IMU/GPS prediction
LOST_S       = 0.5               # no accepted match this long -> pose None (pure_pursuit stops)


def read_pcd(fname):
    """ASCII .pcd -> (N, 3) float32 x, y, z."""
    with open(fname, "rb") as f:
        hdr = {}
        while "DATA" not in hdr:
            w = f.readline().decode("ascii").split()
            if w:
                hdr[w[0]] = w[1:]
        if hdr["DATA"][0] != "ascii":
            # ponytail: ascii only; binary PCD = np.frombuffer with a dtype built from FIELDS/SIZE/TYPE
            raise ValueError(f"{fname}: PCD DATA {hdr['DATA'][0]} not supported (save it as ascii)")
        return np.loadtxt(f, dtype=np.float32, usecols=[hdr["FIELDS"].index(k) for k in "xyz"], ndmin=2)


def _voxel(p, size):
    _, i = np.unique(np.floor(p / size).astype(np.int64), axis=0, return_index=True)
    return p[i]


def map_points(pts):
    """Map cloud -> 2D structure points (MIN_H..MAX_H above the lowest point of their ground cell)."""
    _, cell = np.unique(np.floor(pts[:, :2] / GROUND_CELL_M).astype(np.int64), axis=0, return_inverse=True)
    cell = cell.ravel()
    ground = np.full(cell.max() + 1, np.inf, np.float32)
    np.minimum.at(ground, cell, pts[:, 2])
    h = pts[:, 2] - ground[cell]
    return _voxel(pts[(h > MIN_H) & (h < MAX_H), :2], MAP_VOXEL_M)


def scan_points(fwd, left, z):
    """Velodyne scan (car frame, z above ground) -> 2D structure points."""
    r = np.hypot(fwd - velodyne.LIDAR_FWD_M, left)
    m = (z > MIN_H) & (z < MAX_H) & (r > MIN_RANGE_M) & (r < MAX_RANGE_M)
    return _voxel(np.column_stack([fwd[m], left[m]]), SCAN_VOXEL_M)


def icp(tree, ref, scan, pose, dists):
    """2D point-to-point ICP of scan (car frame) onto ref (map) from pose (x, y, yaw).
    -> (pose, inlier fraction within the last radius)."""
    # ponytail: point-to-point; point-to-plane converges better along long featureless walls
    x, y, th = pose
    fit = 0.0
    for md in dists:
        for _ in range(ICP_ITERS):
            c, s = math.cos(th), math.sin(th)
            p = scan @ np.array([[c, s], [-s, c]]) + (x, y)
            d, j = tree.query(p, distance_upper_bound=md, workers=-1)
            m = np.isfinite(d)
            fit = m.mean()
            if m.sum() < 20:
                return (x, y, th), 0.0
            a, b = p[m], ref[j[m]]
            ca, cb = a.mean(0), b.mean(0)
            H = (a - ca).T @ (b - cb)
            dth = math.atan2(H[0, 1] - H[1, 0], H[0, 0] + H[1, 1])
            c, s = math.cos(dth), math.sin(dth)
            x, y = c * (x - ca[0]) - s * (y - ca[1]) + cb[0], s * (x - ca[0]) + c * (y - ca[1]) + cb[1]
            th += dth
    return (x, y, math.atan2(math.sin(th), math.cos(th))), fit


class LidarPose:
    def __init__(self, pcd, lidar, motion=None):
        """lidar: velodyne.Velodyne. motion: gnss_pose.FusedPose (gyro_yaw + GPS speed) or None."""
        t0 = time.time()
        self.ref = map_points(read_pcd(pcd))
        self.tree = cKDTree(self.ref)
        self.load_s = time.time() - t0
        self.lidar, self.motion = lidar, motion
        self.est = None                      # (t, x, y, yaw, gyro_yaw at t) of the last accepted match
        self.fit = 0.0

    def _scan(self, timeout):
        if not self.lidar.new.wait(timeout):
            return None
        self.lidar.new.clear()
        t, *pts = self.lidar.scan
        return t, scan_points(*pts)

    def _gyro(self):
        return self.motion.gyro_yaw if self.motion else 0.0

    def _predict(self, t):
        t0, x, y, yaw, g0 = self.est
        dyaw = self._gyro() - g0
        dist = (self.motion.speed if self.motion else 0.0) * max(0.0, t - t0)
        mid = yaw + dyaw / 2
        return x + dist * math.cos(mid), y + dist * math.sin(mid), yaw + dyaw

    def localize(self, guess):
        """First match around guess (x, y, yaw), trying a grid of positions x headings. Starts tracking
        on success.
        -> inlier fraction (>= MIN_FIT means localized)."""
        s = self._scan(2.0)
        if s is None:
            return 0.0
        t, pts = s
        hyps = [(guess[0] + dx, guess[1] + dy, guess[2] + math.radians(a))
                for dx in INIT_XY_M for dy in INIT_XY_M for a in INIT_YAWS_DEG]
        pose, self.fit = max((icp(self.tree, self.ref, pts, h, INIT_DISTS) for h in hyps), key=lambda r: r[1])
        if self.fit >= MIN_FIT:
            self.est = (t, *pose, self._gyro())
            threading.Thread(target=self._run, daemon=True).start()
        return self.fit

    def _run(self):
        while True:
            s = self._scan(1.0)
            if s is None:
                continue
            t, pts = s
            pred = self._predict(t)
            pose, self.fit = icp(self.tree, self.ref, pts, pred, TRACK_DISTS)
            if self.fit >= MIN_FIT and math.hypot(pose[0] - pred[0], pose[1] - pred[1]) < MAX_JUMP_M:
                self.est = (t, *pose, self._gyro())

    def pose(self):
        e = self.est
        if e is None or time.monotonic() - e[0] > LOST_S:
            return None
        return self._predict(time.monotonic())

    def fix_name(self):
        return f"ICP {self.fit:.2f}"

    def close(self):
        if self.motion:
            self.motion.close()
