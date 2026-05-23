import pybullet as p
import numpy as np
import math

# ---------------------------------------------------------------------------
# Track layout:
#   OUTER wall = rounded rectangle (straight edges, circular corners)
#   INNER wall = ellipse (continuously curved)
#
# This asymmetry means each corner has a different shape on the inside
# vs outside — the inner wall curves away faster at entry than exit,
# naturally producing a wider entry / tighter exit racing line.
# ---------------------------------------------------------------------------
STRAIGHT_LEN   = 14.0   # half-length of outer straight
CORNER_RADIUS  = 5.0    # outer corner radius
TRACK_WIDTH    = 6.0    # approximate track width on the straights
INNER_A        = 10.0   # inner ellipse half-width  (x)
INNER_B        = 3.5    # inner ellipse half-height (y) — tighter than outer
WALL_HEIGHT    = 0.8
WALL_THICKNESS = 0.25
N_CORNER       = 28     # segments per corner arc
N_ELLIPSE      = 80     # segments for inner ellipse


def build_centreline(n_points=300):
    """
    Centreline runs between the outer rounded rectangle and inner ellipse.
    Approximated as a rounded rectangle slightly smaller than the outer wall.
    """
    cr  = CORNER_RADIUS - TRACK_WIDTH / 2.0
    sl  = STRAIGHT_LEN  - TRACK_WIDTH / 4.0
    pts = []

    # Bottom straight
    for t in np.linspace(-sl, sl, 50, endpoint=False):
        pts.append((t, -(cr + TRACK_WIDTH / 2.0)))

    # Right hairpin
    for t in np.linspace(-math.pi/2, math.pi/2, 36, endpoint=False):
        pts.append((sl + cr * math.cos(t), cr * math.sin(t)))

    # Top straight
    for t in np.linspace(sl, -sl, 50, endpoint=False):
        pts.append((t, (cr + TRACK_WIDTH / 2.0)))

    # Left hairpin
    for t in np.linspace(math.pi/2, 3*math.pi/2, 36, endpoint=False):
        pts.append((-sl + cr * math.cos(t), cr * math.sin(t)))

    pts   = np.array(pts)
    n     = len(pts)
    dists = np.zeros(n)
    for i in range(1, n):
        dists[i] = dists[i-1] + np.linalg.norm(pts[i] - pts[i-1])
    total   = dists[-1] + np.linalg.norm(pts[0] - pts[-1])
    targets = np.linspace(0, total, n_points, endpoint=False)
    out     = []
    for t in targets:
        t_mod = t % total
        idx   = max(0, np.searchsorted(dists, t_mod, side='right') - 1)
        nxt   = (idx + 1) % n
        seg   = (total - dists[-1]) if nxt == 0 else (dists[nxt] - dists[idx])
        frac  = np.clip((t_mod - dists[idx]) / seg, 0.0, 1.0) if seg > 1e-9 else 0.0
        out.append(pts[idx] + frac * (pts[nxt] - pts[idx]))
    return np.array(out)


def build_arc_lengths(centreline):
    n   = len(centreline)
    arc = np.zeros(n)
    for i in range(1, n):
        arc[i] = arc[i-1] + np.linalg.norm(centreline[i] - centreline[i-1])
    return arc


def project_onto_centreline(x, y, centreline, arc_lengths):
    pt        = np.array([x, y])
    n         = len(centreline)
    total_len = arc_lengths[-1] + np.linalg.norm(centreline[0] - centreline[-1])
    best_s, best_dist, best_pt, best_idx = 0.0, float('inf'), centreline[0], 0
    for i in range(n):
        j      = (i + 1) % n
        a, b   = centreline[i], centreline[j]
        ab     = b - a
        ab_len = np.linalg.norm(ab)
        if ab_len < 1e-9:
            continue
        t    = np.clip(np.dot(pt - a, ab) / ab_len**2, 0.0, 1.0)
        proj = a + t * ab
        dist = np.linalg.norm(pt - proj)
        if dist < best_dist:
            best_dist = dist
            best_pt   = proj
            best_idx  = i
            seg_total = (total_len - arc_lengths[-1]) if j == 0 \
                        else (arc_lengths[j] - arc_lengths[i])
            best_s    = arc_lengths[i] + t * seg_total
    return best_s, best_pt, best_idx


# ---------------------------------------------------------------------------
# Wall helpers
# ---------------------------------------------------------------------------
def _box_wall(client, cx, cy, cz, hx, hy, hz, yaw, color):
    orn = client.getQuaternionFromEuler([0, 0, yaw])
    col = client.createCollisionShape(p.GEOM_BOX, halfExtents=[hx, hy, hz])
    vis = client.createVisualShape(p.GEOM_BOX, halfExtents=[hx, hy, hz],
                                   rgbaColor=color)
    return client.createMultiBody(baseMass=0,
                                  baseCollisionShapeIndex=col,
                                  baseVisualShapeIndex=vis,
                                  basePosition=[cx, cy, cz],
                                  baseOrientation=orn)


def _build_straight_wall(client, x_start, x_end, y, color):
    cx = (x_start + x_end) / 2.0
    hx = abs(x_end - x_start) / 2.0 + 0.1
    return [_box_wall(client, cx, y, WALL_HEIGHT/2,
                      hx, WALL_THICKNESS/2, WALL_HEIGHT/2, 0, color)]


def _build_arc_wall(client, cx_arc, cy_arc, radius,
                    start_angle, end_angle, n_seg, color):
    wall_ids = []
    angles   = np.linspace(start_angle, end_angle, n_seg + 1)
    for i in range(n_seg):
        a0, a1 = angles[i], angles[i+1]
        x0 = cx_arc + radius * math.cos(a0)
        y0 = cy_arc + radius * math.sin(a0)
        x1 = cx_arc + radius * math.cos(a1)
        y1 = cy_arc + radius * math.sin(a1)
        seg_len = math.sqrt((x1-x0)**2 + (y1-y0)**2) + 0.05
        yaw     = math.atan2(y1-y0, x1-x0)
        wall_ids.append(_box_wall(
            client, (x0+x1)/2, (y0+y1)/2, WALL_HEIGHT/2,
            seg_len/2, WALL_THICKNESS/2, WALL_HEIGHT/2, yaw, color))
    return wall_ids


def _build_ellipse_wall(client, a, b, n_seg, color):
    """Inner wall as a pure ellipse — continuously varying curvature."""
    wall_ids = []
    angles   = np.linspace(0, 2*math.pi, n_seg, endpoint=False)
    for i in range(n_seg):
        a0 = angles[i]
        a1 = angles[(i+1) % n_seg]
        x0, y0 = a * math.cos(a0), b * math.sin(a0)
        x1, y1 = a * math.cos(a1), b * math.sin(a1)
        seg_len = math.sqrt((x1-x0)**2 + (y1-y0)**2) + 0.04
        yaw     = math.atan2(y1-y0, x1-x0)
        wall_ids.append(_box_wall(
            client, (x0+x1)/2, (y0+y1)/2, WALL_HEIGHT/2,
            seg_len/2, WALL_THICKNESS/2, WALL_HEIGHT/2, yaw, color))
    return wall_ids


class Track:
    """
    Hybrid track: rounded-rectangle outer wall + ellipse inner wall.

    The outer wall has straight sections with circular corners (constant
    curvature per section). The inner wall is a continuous ellipse with
    varying curvature — tightest at the top/bottom, gentler on the sides.

    At each corner the inner wall curves away differently on entry vs exit,
    which forces the agent to take a wider entry and tighter exit line —
    exactly the classic racing line pattern.
    """

    def __init__(self, client):
        self.client       = client
        self.centreline   = build_centreline(300)
        self.arc_lengths  = build_arc_lengths(self.centreline)
        self.total_length = (self.arc_lengths[-1] +
                             np.linalg.norm(self.centreline[0] - self.centreline[-1]))
        self._build_walls()

    def _build_walls(self):
        c    = self.client
        RED  = (0.8, 0.1, 0.1, 1)
        BLUE = (0.1, 0.1, 0.8, 1)
        OR   = CORNER_RADIUS + TRACK_WIDTH / 2.0   # outer wall radius

        # --- Outer wall: rounded rectangle ---
        _build_straight_wall(c, -STRAIGHT_LEN, STRAIGHT_LEN, -(OR), RED)
        _build_straight_wall(c, -STRAIGHT_LEN, STRAIGHT_LEN,  (OR), RED)
        _build_arc_wall(c,  STRAIGHT_LEN, 0, OR, -math.pi/2, math.pi/2,  N_CORNER, RED)
        _build_arc_wall(c, -STRAIGHT_LEN, 0, OR,  math.pi/2, 3*math.pi/2, N_CORNER, RED)

        # --- Inner wall: ellipse ---
        _build_ellipse_wall(c, INNER_A, INNER_B, N_ELLIPSE, BLUE)

    # ------------------------------------------------------------------
    def get_progress(self, x, y):
        s, _, _ = project_onto_centreline(x, y, self.centreline, self.arc_lengths)
        return s

    def progress_delta(self, prev_s, curr_s):
        delta = curr_s - prev_s
        half  = self.total_length / 2.0
        if delta >  half: delta -= self.total_length
        if delta < -half: delta += self.total_length
        return delta

    def laps_completed(self, total_distance):
        return int(total_distance / self.total_length)

    def point_on_track(self, x, y):
        pt = np.array([x, y])
        _, closest, _ = project_onto_centreline(
            x, y, self.centreline, self.arc_lengths)
        dist = np.linalg.norm(pt - closest)
        return dist < (TRACK_WIDTH / 2.0 + 0.5)

    def get_wall_distances(self, x, y, yaw, n_rays=7, max_dist=10.0):
        ray_from  = [x, y, 0.2]
        angles    = np.linspace(-math.pi/2, math.pi/2, n_rays)
        distances = []
        for angle in angles:
            d      = yaw + angle
            ray_to = [x + max_dist * math.cos(d),
                      y + max_dist * math.sin(d), 0.2]
            result = self.client.rayTest(ray_from, ray_to)
            distances.append(float(result[0][2]))
        return distances

    @staticmethod
    def get_start_position():
        """Bottom of track, centre, facing right."""
        return 0.0, -(CORNER_RADIUS + TRACK_WIDTH/2.0 - TRACK_WIDTH/2.0), 0.0

    @staticmethod
    def get_width():
        return TRACK_WIDTH