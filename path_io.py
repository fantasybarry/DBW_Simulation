"""Path loading: Lanelet2 .osm map -> local (x, y[, mph]) waypoints + the lat/lon origin.

The route's centerline (each lanelet's `centerline` way, else the mean of its left/right bounds)
becomes the path. x = east, y = north, meters from the origin; gnss_pose maps live GPS with the
same ll_to_xy, so the path and the car share one frame.
"""
import math
import xml.etree.ElementTree as ET

R_EARTH_M  = 6378137.0
SPACING_M  = 1.0          # centerline resample spacing
KMH_TO_MPH = 0.621371     # Autoware's lanelet speed_limit tag is km/h


def ll_to_xy(lat, lon, lat0, lon0):
    """Meters east/north of (lat0, lon0)."""
    # ponytail: equirectangular; path and GPS share the same distortion, so tracking is unaffected
    # within a few km. Switch to UTM if routes span tens of km.
    return (math.radians(lon - lon0) * R_EARTH_M * math.cos(math.radians(lat0)),
            math.radians(lat - lat0) * R_EARTH_M)


def _d(a, b):
    return math.hypot(b[0] - a[0], b[1] - a[1])


def _resample(pts, n):
    """n points evenly spaced by arc length along polyline pts."""
    cum = [0.0]
    for a, b in zip(pts, pts[1:]):
        cum.append(cum[-1] + _d(a, b))
    out, j = [], 0
    for k in range(n):
        s = cum[-1] * k / (n - 1)
        while j < len(pts) - 2 and cum[j + 1] < s:
            j += 1
        t = (s - cum[j]) / ((cum[j + 1] - cum[j]) or 1e-9)
        (ax, ay), (bx, by) = pts[j], pts[j + 1]
        out.append((ax + t * (bx - ax), ay + t * (by - ay)))
    return out


def _centerline(mem):
    if "centerline" in mem:
        return mem["centerline"]
    left, right = mem["left"], mem["right"]
    if _d(left[0], right[-1]) + _d(left[-1], right[0]) < _d(left[0], right[0]) + _d(left[-1], right[-1]):
        right = right[::-1]                                       # bounds stored in opposite directions
    n = max(len(left), len(right), round(sum(_d(a, b) for a, b in zip(left, left[1:])) / SPACING_M) + 1)
    return [((a[0] + b[0]) / 2, (a[1] + b[1]) / 2) for a, b in zip(_resample(left, n), _resample(right, n))]


def _mph(limit):
    if not limit:
        return None
    v = float(limit.split()[0])
    return v if "mph" in limit else v * KMH_TO_MPH


def load_path(fname, route=None):
    """-> (origin, [(x, y[, mph]), ...]). mph = the lanelet's speed_limit, if every route lanelet has one.
    Nodes with local_x/local_y tags (Autoware maps): those coordinates as-is -- the point cloud map's
    frame -- and origin None. Otherwise lat/lon -> meters from origin = (lat, lon) of the first node.
    route: lanelet ids in driving order; a single id (or None = the first lanelet with no predecessor)
    is extended by following successors until the route ends, branches or closes a loop."""
    root = ET.parse(fname).getroot()
    nodes = list(root.iter("node"))
    tags = [{t.get("k"): t.get("v") for t in n.iter("tag")} for n in nodes]
    if all("local_x" in t and "local_y" in t for t in tags):
        origin = None
        xy = {n.get("id"): (float(t["local_x"]), float(t["local_y"])) for n, t in zip(nodes, tags)}
    else:
        origin = (float(nodes[0].get("lat")), float(nodes[0].get("lon")))
        xy = {n.get("id"): ll_to_xy(float(n.get("lat")), float(n.get("lon")), *origin) for n in nodes}
    ways = {w.get("id"): [nd.get("ref") for nd in w.iter("nd")] for w in root.iter("way")}
    lanelets = {}                                                 # id -> (bound node ids by role, mph)
    for r in root.iter("relation"):
        rtags = {t.get("k"): t.get("v") for t in r.iter("tag")}
        if rtags.get("type") == "lanelet":
            mem = {m.get("role"): ways[m.get("ref")] for m in r.iter("member") if m.get("type") == "way"}
            lanelets[r.get("id")] = (mem, _mph(rtags.get("speed_limit")))
    if not lanelets:
        raise ValueError(f"{fname}: no lanelets (relations tagged type=lanelet)")
    missing = [i for i in (route or []) if i not in lanelets]
    if missing:
        raise ValueError(f"lanelet(s) {','.join(missing)} not in {fname}")

    # successor = the lanelet whose bounds start at the node ids where this one's end
    by_start = {}
    for i, (m, _) in lanelets.items():
        by_start.setdefault((m["left"][0], m["right"][0]), []).append(i)
    succ = lambda i: by_start.get((lanelets[i][0]["left"][-1], lanelets[i][0]["right"][-1]), [])
    if not route or len(route) == 1:
        has_pred = {j for i in lanelets for j in succ(i)}
        route = route or [next((i for i in lanelets if i not in has_pred), next(iter(lanelets)))]
        while len(succ(route[-1])) == 1 and succ(route[-1])[0] not in route:
            route.append(succ(route[-1])[0])
        if len(succ(route[-1])) > 1:
            raise ValueError(f"route {','.join(route)} branches at {route[-1]} -> {','.join(succ(route[-1]))}; "
                             f"pass the full --route")

    centerline = lambda i: _centerline({role: [xy[n] for n in ids] for role, ids in lanelets[i][0].items()})
    path = []
    for lid in route:
        cl, mph = centerline(lid), lanelets[lid][1]
        # orient each lanelet to continue from the previous one (the first: toward the second)
        if path:
            flip = _d(path[-1], cl[-1]) < _d(path[-1], cl[0])
        elif len(route) > 1:
            nxt = centerline(route[1])
            flip = min(_d(cl[0], nxt[0]), _d(cl[0], nxt[-1])) < min(_d(cl[-1], nxt[0]), _d(cl[-1], nxt[-1]))
        else:
            flip = False
        for p in (cl[::-1] if flip else cl):
            if not path or _d(path[-1], p) > 0.05:                 # drop the shared joint point
                path.append((*p, mph) if mph is not None else p)
    return origin, path
