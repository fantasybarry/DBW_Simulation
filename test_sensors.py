"""Offline checks for the map / GPS / IMU / lidar parsing. Run: python3 test_sensors.py"""
import math
import os
import struct
import tempfile

from path_io import load_path, ll_to_xy

LAT0, LON0 = 40.0, -83.0
M_LAT = 1 / 111319.49                       # deg per meter north
M_LON = M_LAT / math.cos(math.radians(LAT0))


def osm(lanelets):
    """lanelets: {id: (left [(x,y)], right [(x,y)], speed_limit or None)} -> Lanelet2 .osm text."""
    nodes, ways, rels, nid = [], [], [], [1000]
    for lid, (left, right, lim) in lanelets.items():
        wids = []
        for side in (left, right):
            refs = []
            for x, y in side:
                nid[0] += 1
                nodes.append(f'<node id="{nid[0]}" lat="{LAT0 + y * M_LAT}" lon="{LON0 + x * M_LON}"/>')
                refs.append(f'<nd ref="{nid[0]}"/>')
            nid[0] += 1
            wids.append(nid[0])
            ways.append(f'<way id="{nid[0]}">{"".join(refs)}</way>')
        tag = f'<tag k="speed_limit" v="{lim}"/>' if lim else ""
        rels.append(f'<relation id="{lid}"><member type="way" role="left" ref="{wids[0]}"/>'
                    f'<member type="way" role="right" ref="{wids[1]}"/><tag k="type" v="lanelet"/>{tag}</relation>')
    return f'<osm version="0.6">{"".join(nodes + ways + rels)}</osm>'


def test_osm():
    # lanelet 1: east 0->20 m, lane y in [-1.5, 1.5]; lanelet 2: continues east 20->40, its RIGHT bound
    # stored backwards (must be re-oriented); route given in order
    m = osm({"1": ([(0, 1.5), (20, 1.5)], [(0, -1.5), (20, -1.5)], "16"),
             "2": ([(20, 1.5), (40, 1.5)], [(40, -1.5), (20, -1.5)], "16")})
    with tempfile.NamedTemporaryFile("w", suffix=".osm", delete=False) as f:
        f.write(m)
    try:
        origin, path = load_path(f.name, ["1", "2"])
        x0, y0 = ll_to_xy(LAT0, LON0, *origin)                   # map (0,0) in the path frame
        assert len(path) == 41, len(path)                        # 1 m spacing, joint point not duplicated
        assert all(abs(p[1] - y0) < 0.01 for p in path), "centerline must be the lane middle"
        assert abs(path[0][0] - x0) < 0.01 and abs(path[-1][0] - x0 - 40) < 0.05
        assert abs(path[0][2] - 16 * 0.621371) < 1e-6           # km/h speed_limit -> mph
        # reversed route order -> path runs west
        _, back = load_path(f.name, ["2", "1"])
        assert back[0][0] > back[-1][0]
        # no --route: these two don't share bound nodes, so the auto route is just the first lanelet
        assert len(load_path(f.name)[1]) == 21
    finally:
        os.unlink(f.name)
    # Autoware sample map (local_x/y, 16-lanelet one-way loop): auto route = the whole loop, closed
    here = os.path.dirname(os.path.abspath(__file__))
    sample = os.path.join(here, "sample-map-planning", "lanelet2_map.osm")
    if os.path.exists(sample):
        origin, loop = load_path(sample)
        assert origin is None and math.hypot(loop[0][0] - loop[-1][0], loop[0][1] - loop[-1][1]) < 2.0


def nmea(body):
    cs = 0
    for c in body:
        cs ^= ord(c)
    return f"${body}*{cs:02X}"


def test_nmea():
    from gnss_pose import nmea_fields, _deg
    f = nmea_fields(nmea("GNGGA,123519,4807.038,N,01131.000,W,4,08,0.9,545.4,M,46.9,M,,"))
    assert f and f[0] == "$GNGGA" and f[6] == "4"
    assert abs(_deg(f[2], f[3]) - (48 + 7.038 / 60)) < 1e-9
    assert abs(_deg(f[4], f[5]) + (11 + 31.0 / 60)) < 1e-9
    good = nmea("VNYMR,+010.0,+000.0,+000.0,+1.0,+0.0,+0.0,+0.0,+0.0,-9.8,+0.0,+0.0,+0.1")
    assert nmea_fields(good)[12] == "+0.1"
    assert nmea_fields(good[:-1] + ("0" if good[-1] != "0" else "1")) is None   # bad checksum rejected


def test_velodyne():
    import numpy as np
    import velodyne as v
    # one packet: every block at azimuth = +rot correction of laser 5 (0 deg vertical), so laser 5 points
    # dead ahead; laser 5 at 5 m, all others 0 (no return)
    az = round(math.degrees(v._ROT[5]) * 100) % 36000
    block = struct.pack("<HH", 0xEEFF, az) + b"".join(
        struct.pack("<HB", round(5 / v.DIST_RES_M) if k == 5 else 0, 100) for k in range(32))
    pkt = block * 12 + b"\0" * 6
    fwd, left, z = v.parse(pkt)
    assert len(fwd) == 12 and np.allclose(z, v.LIDAR_HEIGHT_M) and np.allclose(fwd, 5 + v.LIDAR_FWD_M) and np.allclose(left, 0, atol=1e-6)
    lid = v.Velodyne.__new__(v.Velodyne)
    lid.scan = (__import__("time").monotonic(), fwd, left, z)
    v.MIN_POINTS = 5
    assert lid.blocked(0.0, 1.0)                                  # dead ahead, straight arc
    assert lid.blocked(0.1, 1.0) is None                          # arc curves ~1.9 m left at 6.2 m: clear
    lid.scan = None
    assert lid.blocked(0.0, 1.0) == "lidar: no data"             # fail safe


def test_icp():
    import numpy as np
    from scipy.spatial import cKDTree
    import lidar_pose as lp
    rng = np.random.default_rng(0)
    # synthetic 3D map: ground plane + 40 random 3 m-tall walls
    walls = []
    for _ in range(40):
        (x0, y0), a, n = rng.uniform(-60, 60, 2), rng.uniform(0, math.pi), int(rng.uniform(40, 150))
        s = np.arange(n) * 0.1
        for zz in np.arange(0, 3, 0.3):
            walls.append(np.column_stack([x0 + s * math.cos(a), y0 + s * math.sin(a), np.full(n, zz)]))
    ground = np.column_stack([rng.uniform(-70, 70, (40000, 2)), rng.normal(0, 0.02, 40000)])
    cloud = np.vstack(walls + [ground]).astype(np.float32)
    ref = lp.map_points(cloud)
    assert len(ref) and not np.any(np.isin(ref, ground[:, :2])), "ground must be dropped"
    # scan: map points seen from the true pose, in the car frame
    tx, ty, tth = 5.0, -3.0, math.radians(30)
    d = cloud[:, :2] - (tx, ty)
    c, s = math.cos(tth), math.sin(tth)
    fwd, left = d[:, 0] * c + d[:, 1] * s, -d[:, 0] * s + d[:, 1] * c
    pts = lp.scan_points(fwd + rng.normal(0, 0.03, len(fwd)), left, cloud[:, 2])
    (x, y, th), fit = lp.icp(cKDTree(ref), ref, pts, (tx + 1.5, ty - 1.0, tth + math.radians(10)), lp.INIT_DISTS)
    assert fit > 0.8 and math.hypot(x - tx, y - ty) < 0.1 and abs(math.degrees(th - tth)) < 1, (x, y, th, fit)


if __name__ == "__main__":
    test_osm()
    test_nmea()
    test_velodyne()
    test_icp()
    print("ok")
