"""Velodyne VLP-32C reader -> "is the pursuit arc ahead blocked?" (obstacle stop).

Reads the raw UDP data packets (port 2368) and turns each ~1-revolution batch into points in the car
frame (forward from the rear axle, left, height above ground) -> self.scan, signalling self.new
(lidar_pose matches these to the point cloud map). blocked() checks a car-width corridor bent along the current pursuit arc. No/stale lidar data
counts as blocked (fail safe).
"""
import socket
import threading
import time

import numpy as np

LIDAR_HEIGHT_M = 1.9      # lidar origin above the ground -- measure
LIDAR_FWD_M    = 1.2      # lidar ahead of the rear axle -- measure
AZ_OFFSET_DEG  = 0.0      # mount yaw: tune until a pole dead ahead reads left = 0
DIST_RES_M     = 0.004    # VLP-32C range unit (4 mm)
CLEAR_FRONT_M  = 3.4      # rear axle -> front bumper (Leaf ~3.45): closer hits are the car itself
HALF_WIDTH_M   = 1.2      # corridor half-width (Leaf is 1.79 m wide, + margin)
MIN_Z_M, MAX_Z_M = 0.3, 2.2   # height band that counts as obstacle (ground/curb below, overhang above)
MIN_POINTS     = 10       # hits needed (rejects rain/dust speckle)
STOP_GAP_M     = 5.0      # always keep this much clear ahead of the bumper...
DECEL_MPS2     = 2.0      # ...plus braking distance v^2 / (2 a)
SCAN_S         = 0.1      # batch window = 1 revolution at the default 600 rpm
STALE_S        = 0.5

# VLP-32C laser id -> (vertical deg, azimuth correction deg), from the VLP-32C manual / ROS VLP32C_db.yaml
_CAL = np.array([
    (-25.0, 1.4), (-1.0, -4.2), (-1.667, 1.4), (-15.639, -1.4), (-11.31, 1.4), (0.0, -1.4),
    (-0.667, 4.2), (-8.843, -1.4), (-7.254, 1.4), (0.333, -4.2), (-0.333, 1.4), (-6.148, -1.4),
    (-5.333, 4.2), (1.333, -1.4), (0.667, 4.2), (-4.0, -1.4), (-4.667, 1.4), (1.667, -4.2),
    (1.0, 1.4), (-3.667, -4.2), (-3.333, 4.2), (3.333, -1.4), (2.333, 1.4), (-2.667, -1.4),
    (-3.0, 1.4), (7.0, -1.4), (4.667, 1.4), (-2.333, -4.2), (-2.0, 4.2), (15.0, -1.4),
    (10.333, 1.4), (-1.333, -1.4)])
_VERT = np.radians(_CAL[:, 0])
_ROT = np.radians(_CAL[:, 1])

PKT = np.dtype([("blocks", [("flag", "<u2"), ("azi", "<u2"), ("ch", [("d", "<u2"), ("r", "u1")], (32,))], (12,)),
                ("ts", "<u4"), ("factory", "u1", (2,))])
assert PKT.itemsize == 1206


def parse(buf):
    """Concatenated 1206-byte packets -> (fwd, left, z) arrays of all returns, car frame (m)."""
    b = np.frombuffer(buf, PKT)["blocks"]
    b = b[b["flag"] == 0xEEFF]                                    # bytes FF EE = valid firing block
    az = np.radians(b["azi"] / 100.0 + AZ_OFFSET_DEG)[..., None] - _ROT   # cw from forward
    r = b["ch"]["d"] * DIST_RES_M
    horiz = r * np.cos(_VERT)
    fwd = horiz * np.cos(az) + LIDAR_FWD_M
    left = -horiz * np.sin(az)
    z = r * np.sin(_VERT) + LIDAR_HEIGHT_M
    m = r > 0
    return fwd[m], left[m], z[m]


class Velodyne:
    def __init__(self, port=2368):
        self.port = port
        self.scan = None                                          # (t, fwd, left, z)
        self.new = threading.Event()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 << 20)
        self.sock.bind(("", port))
        self.sock.settimeout(0.5)
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        buf, t0 = [], time.monotonic()
        while True:
            try:
                data = self.sock.recv(1500)
            except socket.timeout:
                continue                                          # no data -> scan goes stale -> blocked
            except OSError:
                break                                             # closed
            if len(data) == 1206:                                 # data packet (512-byte ones are GPS/position)
                buf.append(data)
            now = time.monotonic()
            if now - t0 >= SCAN_S:
                if buf:
                    self.scan = (now, *parse(b"".join(buf)))
                    self.new.set()
                buf, t0 = [], now

    def blocked(self, kappa, speed_mps):
        """Reason string if the corridor along the pursuit arc (curvature kappa, + = left) is blocked
        within stopping distance, else None."""
        scan = self.scan
        if scan is None or time.monotonic() - scan[0] > STALE_S:
            return "lidar: no data"
        _, fwd, left, z = scan
        reach = CLEAR_FRONT_M + STOP_GAP_M + speed_mps ** 2 / (2 * DECEL_MPS2)
        centre = 0.5 * kappa * fwd ** 2                          # arc's lateral offset at that range
        hit = ((fwd > CLEAR_FRONT_M) & (fwd < reach) & (z > MIN_Z_M) & (z < MAX_Z_M)
               & (np.abs(left - centre) < HALF_WIDTH_M))
        if hit.sum() < MIN_POINTS:
            return None
        return f"obstacle {fwd[hit].min() - CLEAR_FRONT_M:.1f} m ahead"

    def close(self):
        self.sock.close()
