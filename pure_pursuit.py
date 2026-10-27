#!/usr/bin/env python3
"""Pure-pursuit path tracker (Jetson side). Feeds steering angle (+ target speed) to the DBW
over the existing 0x740 path -- NO firmware change; the closed-loop angle loop does the steering.

    python3 pure_pursuit.py --sim --path map.osm --route 101,102      # offline: simulate the map route
    python3 pure_pursuit.py --drive --path map.osm --route 101,102 --imu-port /dev/ttyUSB0
        # on car: GPS + IMU pose (gnss_pose), Velodyne obstacle stop (velodyne), DBW (dbw_serial)
    python3 pure_pursuit.py --drive --path sample-map-planning --imu-port /dev/ttyUSB0
        # Autoware map dir: pose = lidar ICP on pointcloud_map.pcd (lidar_pose) + IMU/GPS-speed prediction

Pure pursuit: pick a point on the path a lookahead-distance Ld ahead, steer toward it.
    alpha = angle from heading to that point
    road-wheel angle = atan2(2 * L * sin(alpha), Ld)
    steering-wheel angle = road-wheel angle * STEER_RATIO
"""
import argparse
import csv
import math
import os
import sys
import time

from path_io import load_path

# ===================== vehicle =====================
WHEELBASE_M = 2.705
STEER_RATIO = 17.0
STEER_SIGN  = -1          # pursuit math is ENU (+ = LEFT); DBW 0x740 is + = RIGHT -> must be -1
STEER_LIMIT_DEG = 180.0   # clamp commanded steering-wheel angle

# ===================== pure pursuit =====================
LOOKAHEAD_M   = 7.0       # base lookahead distance Ld (bigger = smoother, but cuts corners more)
LOOKAHEAD_K   = 0.4       # speed term: Ld = LOOKAHEAD_M + LOOKAHEAD_K * speed_mps
STEER_SMOOTH  = 0.0       # output low-pass (EMA) on steering. KEEP 0: lag in a feedback steering
                          # loop causes weaving/oscillation (even on straights). Smooth via lookahead.
STEER_RATE_MAX = 0.0      # hard per-tick rate cap (0 = off); same lag/instability caveat as above
TARGET_SPEED_MPH = 5.0    # cap / safety ceiling (and the constant speed in "constant" mode)
GOAL_TOL_M    = 1.5       # within this of the last waypoint -> done (stop)
RATE_HZ       = 50

# speed source: "recorded" = the per-waypoint speed (lanelet speed_limit), capped by --speed;
#               "curvature" = slow in curves (see below); "constant" = always --speed.
#               "recorded" falls back to "curvature" when the path has no per-waypoint speed.
SPEED_SOURCE  = "recorded"
OBSTACLE_HOLD_S = 1.0     # stay stopped this long after the lidar corridor clears (no stop-go chatter)
# curvature-based speed: slow in turns so lateral accel stays under A_LAT_MAX.
#   v = sqrt(A_LAT_MAX / |kappa|), clamped to [MIN_SPEED_MPH, cap].  kappa from the pursuit arc.
CURV_SPEED    = True      # False = constant TARGET_SPEED_MPH
A_LAT_MAX     = 1.5       # max lateral accel (m/s^2); lower = slower/gentler in curves
MIN_SPEED_MPH = 1.5       # floor so it doesn't crawl to a stop mid-curve
CURV_WINDOW   = 6         # waypoints ahead to scan for the tightest curve (slows BEFORE it)

# ===================== PATH (edit me): (x, y) waypoints in meters, world frame =====================
PATH = [
    (0.0, 0.0), (10.0, 0.0), (20.0, 2.0), (30.0, 8.0),
    (38.0, 16.0), (42.0, 26.0), (42.0, 36.0),
]
# If PATH was recorded with record_path.py, paste its ORIGIN here so the live GPS shares PATH's
# frame. None -> origin = the first GPS fix at drive time (then PATH must start at that same spot).
ORIGIN_LATLON = None

MPH_TO_MPS = 0.44704


# ---- pose sources -------------------------------------------------------------
# On-car pose comes from gnss_pose.GNSSPose (Septentrio NMEA over USB). Offline uses SimPose.
class SimPose:
    """Offline bicycle-model dead-reckoning, so the pure-pursuit logic can be validated with no car.
    Integrates the commanded steering (DBW convention: + = right) + speed forward."""
    def __init__(self, x, y, heading):
        self.x, self.y, self.yaw = x, y, heading
    def step(self, steer_wheel_deg, speed_mps, dt):
        delta = -math.radians(steer_wheel_deg) / STEER_RATIO    # DBW + = right -> ENU road-wheel angle (+ = left)
        self.x += speed_mps * math.cos(self.yaw) * dt
        self.y += speed_mps * math.sin(self.yaw) * dt
        self.yaw += speed_mps * math.tan(delta) / WHEELBASE_M * dt
    def pose(self):
        return (self.x, self.y, self.yaw)


# ---- pure pursuit core --------------------------------------------------------
def _dist(ax, ay, bx, by):
    return math.hypot(bx - ax, by - ay)

def nearest_idx(x, y, path):
    return min(range(len(path)), key=lambda i: _dist(x, y, path[i][0], path[i][1]))

def lookahead_point(x, y, path, Ld, start_i):
    """INTERPOLATED lookahead: the point where a circle of radius Ld around (x,y) crosses the path
    polyline, searching forward from start_i. Moves continuously along segments (no waypoint-to-
    waypoint hopping), so the commanded steering is smooth instead of stepped."""
    for i in range(start_i, len(path) - 1):
        ax, ay = path[i]; bx, by = path[i + 1]
        dx, dy = bx - ax, by - ay
        fx, fy = ax - x, ay - y
        a = dx * dx + dy * dy
        if a < 1e-9:
            continue
        b = 2.0 * (fx * dx + fy * dy)
        c = fx * fx + fy * fy - Ld * Ld
        disc = b * b - 4 * a * c
        if disc < 0:
            continue
        disc = math.sqrt(disc)
        for t in ((-b + disc) / (2 * a), (-b - disc) / (2 * a)):   # prefer the forward root
            if 0.0 <= t <= 1.0:
                return (ax + t * dx, ay + t * dy), i
    # no crossing ahead: near the end -> the goal; otherwise (car > Ld off the path) -> next waypoint,
    # NEVER the far-away goal (that would cut straight across the rest of the route)
    gx, gy = path[-1]
    if _dist(x, y, gx, gy) <= Ld or start_i >= len(path) - 2:
        return path[-1], len(path) - 1
    return path[start_i + 1], start_i + 1

def _seg_dist(x, y, path, j):
    ax, ay = path[j]; bx, by = path[j + 1]
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy or 1e-9
    t = max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / L2))
    return _dist(x, y, ax + t * dx, ay + t * dy)

def progress_idx(x, y, path, prev, window=8):
    """Segment the car is on: nearest segment in a short window FORWARD of the last one. Forward-
    only + windowed, so a return leg a few meters away (U-turn / loop) can never be snapped onto."""
    hi = min(prev + window, len(path) - 1)
    return min(range(prev, max(hi, prev + 1)), key=lambda j: _seg_dist(x, y, path, j))

def cross_track(x, y, path, i):
    """Signed-ish distance to the nearest path segment (for reporting)."""
    j = max(0, min(i, len(path) - 2))
    ax, ay = path[j]; bx, by = path[j + 1]
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy or 1e-9
    t = max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / L2))
    px, py = ax + t * dx, ay + t * dy
    return _dist(x, y, px, py)

def pursuit_steer(x, y, yaw, path, speed_mps, start_i):
    """start_i: the car's progress segment from the last tick. Returns the updated one as i."""
    Ld = LOOKAHEAD_M + LOOKAHEAD_K * speed_mps
    i = progress_idx(x, y, path, start_i)                        # search from where the CAR is,
    (tx, ty), _ = lookahead_point(x, y, path, Ld, i)             # not from the last target point
    alpha = math.atan2(ty - y, tx - x) - yaw
    alpha = math.atan2(math.sin(alpha), math.cos(alpha))        # wrap to [-pi, pi]
    kappa = 2.0 * math.sin(alpha) / Ld                           # curvature of the pursuit arc (1/m)
    delta = math.atan(kappa * WHEELBASE_M)                        # road-wheel angle (rad)
    steer_deg = STEER_SIGN * math.degrees(delta) * STEER_RATIO
    steer_deg = max(-STEER_LIMIT_DEG, min(STEER_LIMIT_DEG, steer_deg))
    return steer_deg, i, (tx, ty), kappa


def path_curvature_ahead(path, i, window):
    """Max path curvature (1/m) over the next `window` waypoints (menger curvature of triples).
    Uses the actual path geometry, so it slows BEFORE a curve regardless of the lookahead."""
    kmax = 0.0
    n = len(path)
    for j in range(max(1, i), min(i + window, n - 1)):
        ax, ay = path[j - 1]; bx, by = path[j]; cx, cy = path[j + 1]
        a = math.hypot(bx - cx, by - cy); b = math.hypot(ax - cx, ay - cy); c = math.hypot(ax - bx, ay - by)
        if a * b * c < 1e-6:
            continue
        area = abs((bx - ax) * (cy - ay) - (cx - ax) * (by - ay)) * 0.5
        kmax = max(kmax, 4.0 * area / (a * b * c))               # 1/R
    return kmax


def speed_for_curvature(kappa, cap_mph):
    """Target speed (mph) that keeps lateral accel <= A_LAT_MAX on a curve of this curvature."""
    if not CURV_SPEED:
        return cap_mph
    k = abs(kappa)
    if k < 1e-4:
        return cap_mph                                            # effectively straight
    v_mps = math.sqrt(A_LAT_MAX / k)                             # v^2 * kappa = a_lat
    v_mph = v_mps / MPH_TO_MPS
    return max(MIN_SPEED_MPH, min(cap_mph, v_mph))


def main():
    ap = argparse.ArgumentParser(description="Pure-pursuit path tracker.")
    ap.add_argument("--sim", action="store_true", help="offline simulation (no hardware)")
    ap.add_argument("--drive", action="store_true", help="stream to the DBW on the car")
    ap.add_argument("--speed", type=float, default=TARGET_SPEED_MPH)
    ap.add_argument("--no-cap", action="store_true",
                    help="replay the recorded speeds as driven (cap = the path's own max; ignores --speed)")
    ap.add_argument("--port", default=None, help="DBW adapter serial port")
    ap.add_argument("--gps-port", default=None, help="Septentrio GPS serial port (NMEA)")
    ap.add_argument("--imu-port", default=None, help="IMU serial port (enables GPS+IMU fusion)")
    ap.add_argument("--fused", action="store_true", help="GPS+IMU fused pose (heading at standstill)")
    ap.add_argument("--heading", type=float, default=None,
                    help="--fused starting heading, compass deg (cw from N). Default: path start direction "
                         "if parked within 3 m of the path start")
    ap.add_argument("--path", default="lanelet2_map.osm",
                    help="Lanelet2 .osm map, or an Autoware map dir (lanelet2_map.osm + pointcloud_map.pcd)")
    ap.add_argument("--route", default=None,
                    help="lanelet ids in driving order, comma-separated; one id = follow successors from it. "
                         "Default: follow successors from the first lanelet")
    ap.add_argument("--pcd", default=None,
                    help="point cloud map for lidar localization (default: pointcloud_map.pcd beside the .osm)")
    ap.add_argument("--init", default=None,
                    help="lidar localization start guess X,Y,YAW_DEG (map frame). Default: the route start")
    ap.add_argument("--lidar-port", type=int, default=2368, help="Velodyne UDP data port")
    ap.add_argument("--no-lidar", action="store_true", help="drive WITHOUT the lidar obstacle stop")
    ap.add_argument("--log", default=None,
                    help="CSV of the driven track (default with --drive: drive_<time>.csv; plot: plot_track.py)")
    args = ap.parse_args()
    if not (args.sim or args.drive):
        args.sim = True   # default to safe offline sim

    # path + origin: load from record_path.py's file if present, else the built-in demo path
    origin, raw = ORIGIN_LATLON, PATH
    if os.path.isdir(args.path):
        args.path = os.path.join(args.path, "lanelet2_map.osm")
    if args.path and os.path.exists(args.path):
        try:
            origin, raw = load_path(args.path, args.route.split(",") if args.route else None)
        except ValueError as e:
            sys.exit(str(e))
        print(f"loaded {len(raw)} waypoints (origin {origin}) from {args.path}")
    else:
        print(f"no {args.path} -- using the built-in demo PATH ({len(raw)} waypoints)")

    # split into geometry (x,y) and optional recorded speed (mph) per waypoint
    path = [(p[0], p[1]) for p in raw]
    speeds = [p[2] if len(p) > 2 else None for p in raw]
    has_speed = all(s is not None for s in speeds)
    eff_src = SPEED_SOURCE if (SPEED_SOURCE != "recorded" or has_speed) else "curvature"
    print(f"speed source: {eff_src}" + ("" if has_speed else "  (path has no recorded speed)"))
    if eff_src == "recorded" and args.no_cap:
        args.speed = max(speeds)                                  # as driven: never clip a recorded speed
        print(f"  no cap: replaying recorded speeds as driven (max {args.speed:.1f} mph)")
    elif args.no_cap:
        print(f"  --no-cap ignored: path has no recorded speed -> constant {args.speed:.1f} mph")
    if eff_src == "recorded":
        over = sum(s > args.speed for s in speeds)
        if over:
            print(f"  WARNING: recorded up to {max(speeds):.1f} mph, but --speed caps it at {args.speed:.1f} mph "
                  f"({over}/{len(speeds)} waypoints clipped). Pass --speed {max(speeds):.0f} to replay it.")

    dt = 1.0 / RATE_HZ
    gx, gy = path[-1]

    dbw = None
    loc = None
    lidar = None
    pcd = args.pcd or os.path.join(os.path.dirname(args.path), "pointcloud_map.pcd")
    if args.drive:
        from dbw_serial import DBW
        dbw = DBW(args.port)
        if not args.no_lidar:
            from velodyne import Velodyne
            lidar = Velodyne(args.lidar_port)
        fused = args.fused or args.imu_port
        if lidar and os.path.exists(pcd):
            # point cloud map: lidar ICP gives the pose; IMU yaw rate + GPS speed predict between scans
            from lidar_pose import LidarPose, MIN_FIT
            motion = None
            if args.imu_port:
                from gnss_pose import FusedPose
                motion = FusedPose(args.gps_port, args.imu_port)
            print(f"    loading {pcd} ...")
            loc = LidarPose(pcd, lidar, motion)
            srcname = f"lidar ICP on {os.path.basename(pcd)} ({len(loc.ref)} map pts, {loc.load_s:.0f} s)" + \
                      (f" + IMU {motion.imu_port} + GPS {motion.port} speed" if motion else " (no IMU: --imu-port)")
            if args.init:
                guess = [float(v) for v in args.init.split(",")]
                guess[2] = math.radians(guess[2])
            else:
                guess = (*path[0], math.atan2(path[1][1] - path[0][1], path[1][0] - path[0][0]))
            fit = loc.localize(guess)
            if fit < MIN_FIT:
                sys.exit(f"refusing to drive: lidar scan doesn't match the map near x={guess[0]:.1f} "
                         f"y={guess[1]:.1f} (fit {fit:.2f} < {MIN_FIT}). Park at the route start facing along "
                         f"it, or pass --init X,Y,YAW_DEG (map frame, yaw CCW from +x).")
            x0, y0, yaw0 = loc.pose()
            print(f"    localized x={x0:.2f} y={y0:.2f} yaw={math.degrees(yaw0):.1f}  fit={fit:.2f}")
        elif origin is None and os.path.exists(args.path):
            sys.exit("this map has no lat/lon (Local projector), so GPS can't place the car on it: "
                     f"needs the lidar and {pcd}")
        elif fused:
            from gnss_pose import FusedPose
            loc = FusedPose(args.gps_port, args.imu_port)
            srcname = f"GPS {loc.port} + IMU {loc.imu_port} (fused)"
        else:
            from gnss_pose import GNSSPose
            loc = GNSSPose(args.gps_port)
            srcname = f"GPS {loc.port}"
        srcname += f", lidar UDP {lidar.port}" if lidar else ", NO LIDAR STOP"
        print(f">>> DRIVING on {dbw.port}, {srcname}. wheel moves, car drives. e-stop in reach. <<<")
        if not hasattr(loc, "localize"):                    # GPS pose: wait for a fix, tie it to the map
            print("    waiting for fix...")
            while not loc.ready():
                time.sleep(0.2)
            loc.set_origin(*origin) if origin else loc.set_origin()
            print(f"    fix={loc.fix_name()}  origin lat0={loc.lat0:.7f} lon0={loc.lon0:.7f}")
            if loc.fix < 4:
                print(f"    NOTE: fix is {loc.fix_name()} (not RTK) -> position ~1-2 m.")
            if fused:
                msg = loc.seed_from_path(path, args.heading)
                if not msg:
                    sys.exit("refusing to drive: no starting heading -- the car is > 3 m from the path start. "
                             "Move it to the start facing along the path, or pass --heading DEG (compass).")
                print(f"    {msg}")
            else:
                print("    (GPS-only heading -> start rolling straight; use --fused for standstill heading)")

    # initial pose: sim starts on the first segment heading along it; real comes from GNSS
    if args.sim:
        hx = math.atan2(path[1][1] - path[0][1], path[1][0] - path[0][0])
        sim = SimPose(path[0][0], path[0][1], hx)

    print(f"pure pursuit: {len(path)} waypoints, target {args.speed:.0f} mph, "
          f"Ld={LOOKAHEAD_M:.1f}+{LOOKAHEAD_K:.1f}*v, {'SIM' if args.sim else 'DRIVE'}")
    i = 0
    max_cte = 0.0
    log_name = args.log or (time.strftime("drive_%Y%m%d_%H%M%S.csv") if args.drive else None)
    log_f = open(log_name, "w", newline="") if log_name else None
    log_w = csv.writer(log_f) if log_f else None
    if log_w:
        log_w.writerow(["t", "x_m", "y_m", "yaw_deg", "steer_cmd_deg", "speed_cmd_mph", "cte_m", "wp", "fix",
                        "path_file"])
        print(f"logging the driven track -> {log_name}  (plot: python3 plot_track.py {log_name})")
    t_start = time.time()
    ticks = 0
    try:
        last_print = 0.0
        cur_speed_mps = 0.0                                  # last commanded speed (for the lookahead term)
        steer_filt = 0.0                                     # EMA-smoothed steering
        steer_cmd = 0.0                                      # final command after rate cap
        hold_until = 0.0                                     # lidar stop held until this time
        while True:
            if args.sim:
                x, y, yaw = sim.pose()
            else:
                pz = loc.pose()
                if pz is None:                              # no fix / stale GPS or IMU -> stop
                    dbw.steer(steer_cmd, accel=0.0)
                    time.sleep(dt); continue
                x, y, yaw = pz
            if i >= len(path) - 3 and _dist(x, y, gx, gy) <= GOAL_TOL_M:   # i: a closed loop starts AT the goal
                print(f"  reached goal (within {GOAL_TOL_M} m). stopping.")
                break
            steer_deg, i, tgt, kappa = pursuit_steer(x, y, yaw, path, cur_speed_mps, i)
            if eff_src == "recorded":
                speed_mph = min(speeds[i], args.speed)           # replay recorded speed (where the car is), --speed = cap
                if speed_mph < MIN_SPEED_MPH and _dist(x, y, gx, gy) > 3 * GOAL_TOL_M:
                    speed_mph = MIN_SPEED_MPH                    # don't stall mid-path on a recorded ~0 (launch/glitch)
            elif eff_src == "curvature":
                speed_mph = speed_for_curvature(path_curvature_ahead(path, i, CURV_WINDOW), args.speed)
            else:
                speed_mph = args.speed
            why = lidar.blocked(kappa, cur_speed_mps) if lidar else None
            if why:
                hold_until = time.time() + OBSTACLE_HOLD_S
            if time.time() < hold_until:
                speed_mph = 0.0                              # obstacle (or lidar dead) on the arc: stop
            cur_speed_mps = speed_mph * MPH_TO_MPS
            # smooth the steering command: EMA low-pass, then a hard per-tick rate cap
            steer_filt = STEER_SMOOTH * steer_filt + (1.0 - STEER_SMOOTH) * steer_deg
            dd = steer_filt - steer_cmd
            if STEER_RATE_MAX > 0 and abs(dd) > STEER_RATE_MAX:
                steer_cmd += math.copysign(STEER_RATE_MAX, dd)
            else:
                steer_cmd = steer_filt
            cte = cross_track(x, y, path, i)
            max_cte = max(max_cte, cte)

            if args.drive:
                dbw.steer(steer_cmd, accel=speed_mph)       # 0x740: smoothed steer + curvature-scaled speed
            else:
                sim.step(steer_cmd, cur_speed_mps, dt)
            if log_w:
                t_log = (time.time() - t_start) if args.drive else ticks * dt   # sim runs faster than real time
                log_w.writerow([f"{t_log:.3f}", f"{x:.3f}", f"{y:.3f}", f"{math.degrees(yaw):.2f}",
                                f"{steer_cmd:.2f}", f"{speed_mph:.2f}", f"{cte:.3f}", i,
                                loc.fix_name() if loc else "SIM", args.path])

            ticks += 1
            now = time.time()
            if args.sim or (now - last_print) > 0.5:        # throttle drive prints to 2 Hz
                last_print = now
                tag = "" if args.sim else f"fix={loc.fix_name():9} "
                print(f"    {tag}x={x:6.1f} y={y:6.1f} yaw={math.degrees(yaw):6.1f}  "
                      f"steer={steer_deg:+6.1f}  v={speed_mph:4.1f}mph  cte={cte:4.2f}m  wp={i}"
                      + (f"  STOP: {why}" if why else ""))
            time.sleep(0 if args.sim else dt)
        print(f"  done. max cross-track error: {max_cte:.2f} m")
    except KeyboardInterrupt:
        print("\naborted")
    except NotImplementedError as e:
        print(f"\nnot implemented: {e}")
    finally:
        if log_f:
            log_f.close()
            print(f"track saved: {log_name}")
        if dbw:
            dbw.close()
            print("released / stopped.")
        if loc:
            loc.close()
        if lidar:
            lidar.close()


if __name__ == "__main__":
    sys.exit(main())
