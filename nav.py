"""Grid navigation + line-of-sight from arena/v2/world.json (standard library only)."""
import heapq
import json
import math
import time

CELL = 4.0
_SQ2 = 1.4142135
_NB = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
       (-1, -1, _SQ2), (1, -1, _SQ2), (-1, 1, _SQ2), (1, 1, _SQ2)]


class Nav:
    def __init__(self, path):
        with open(path) as f:
            w = json.load(f)
        b = w["bounds"]
        self.minx = b["minX"] - 4.0
        self.miny = b["minY"] - 4.0
        self.maxx = b["maxX"]
        self.maxy = b["maxY"]
        self.nx = int((b["maxX"] + 4.0 - self.minx) / CELL) + 1
        self.ny = int((b["maxY"] + 4.0 - self.miny) / CELL) + 1
        n = self.nx * self.ny
        self.solid = bytearray(n)
        self.top = [-1e9] * n
        for p in w["prisms"]:
            if p["kind"] not in ("building", "stadium"):
                continue
            if p["top"] - p["base"] < 1.5:
                continue
            try:
                self._fill(p)
            except Exception:
                pass
        nx, ny = self.nx, self.ny
        self.block = bytearray(n)
        solid, block = self.solid, self.block
        for j in range(ny):
            row = j * nx
            for i in range(nx):
                if solid[row + i]:
                    for dj in (-1, 0, 1):
                        jj = j + dj
                        if 0 <= jj < ny:
                            for di in (-1, 0, 1):
                                ii = i + di
                                if 0 <= ii < nx:
                                    block[jj * nx + ii] = 1
        # map edge is a wall
        for j in range(ny):
            for i in range(nx):
                cx = self.minx + (i + 0.5) * CELL
                cy = self.miny + (j + 0.5) * CELL
                if cx < b["minX"] + 3 or cx > b["maxX"] - 3 or cy < b["minY"] + 3 or cy > b["maxY"] - 3:
                    block[j * nx + i] = 1

    def _fill(self, p):
        edges = []
        ymin, ymax = 1e18, -1e18
        for r in [p["ring"]] + list(p.get("holes") or []):
            pts = [(r[k], r[k + 1]) for k in range(0, len(r) - 1, 2)]
            m = len(pts)
            for k in range(m):
                edges.append((pts[k], pts[(k + 1) % m]))
        if p["ring"]:
            ys = p["ring"][1::2]
            ymin, ymax = min(ys), max(ys)
        nx, ny = self.nx, self.ny
        jmin = max(0, int((ymin - self.miny) / CELL) - 1)
        jmax = min(ny - 1, int((ymax - self.miny) / CELL) + 1)
        top = p["top"]
        for j in range(jmin, jmax + 1):
            yc = self.miny + (j + 0.5) * CELL
            xs = []
            for (x0, y0), (x1, y1) in edges:
                if (y0 <= yc) != (y1 <= yc):
                    xs.append(x0 + (yc - y0) * (x1 - x0) / (y1 - y0))
            xs.sort()
            for a in range(0, len(xs) - 1, 2):
                i0 = max(0, int(math.ceil((xs[a] - self.minx) / CELL - 0.5)))
                i1 = min(nx - 1, int(math.floor((xs[a + 1] - self.minx) / CELL - 0.5)))
                for i in range(i0, i1 + 1):
                    idx = j * nx + i
                    self.solid[idx] = 1
                    if top > self.top[idx]:
                        self.top[idx] = top

    # ---- helpers
    def cell(self, x, y):
        i = int((x - self.minx) // CELL)
        j = int((y - self.miny) // CELL)
        if i < 0 or j < 0 or i >= self.nx or j >= self.ny:
            return -1
        return j * self.nx + i

    def center(self, idx):
        j, i = divmod(idx, self.nx)
        return (self.minx + (i + 0.5) * CELL, self.miny + (j + 0.5) * CELL)

    def blocked(self, x, y):
        idx = self.cell(x, y)
        return idx < 0 or self.block[idx] == 1

    def clear_seg(self, ax, ay, bx, by):
        """True if the straight walk a->b stays off blocked (inflated) cells."""
        dx, dy = bx - ax, by - ay
        d = math.hypot(dx, dy)
        if d < 1e-6:
            return True
        n = int(d / 2.0) + 1
        for k in range(1, n + 1):
            f = k / n
            px, py = ax + dx * f, ay + dy * f
            if f * d < CELL:  # tolerate being next to a wall at the start
                idx = self.cell(px, py)
                if idx < 0:
                    return False
                continue
            if self.blocked(px, py):
                return False
        return True

    def los(self, ax, ay, bx, by, zref):
        """Line of sight at roughly muzzle height: buildings taller than zref+6 block."""
        dx, dy = bx - ax, by - ay
        d = math.hypot(dx, dy)
        if d < 4.0:
            return True
        n = int(d / 2.0)
        lim = zref + 6.0
        top = self.top
        nx = self.nx
        for k in range(1, n):
            f = k / n
            idx = self.cell(ax + dx * f, ay + dy * f)
            if idx >= 0 and top[idx] > lim:
                return False
        return True

    def nearest_free(self, idx, maxr):
        if idx < 0:
            return -1
        if not self.block[idx]:
            return idx
        j0, i0 = divmod(idx, self.nx)
        for r in range(1, maxr + 1):
            best, bd = -1, 1e9
            for dj in range(-r, r + 1):
                for di in range(-r, r + 1):
                    if max(abs(di), abs(dj)) != r:
                        continue
                    i, j = i0 + di, j0 + dj
                    if 0 <= i < self.nx and 0 <= j < self.ny:
                        k = j * self.nx + i
                        if not self.block[k]:
                            dd = di * di + dj * dj
                            if dd < bd:
                                best, bd = k, dd
            if best >= 0:
                return best
        return -1


class Search:
    """Incremental weighted A*: call step(budget_s) every tick until it returns True."""

    def __init__(self, nav, s, g, weight=1.25, max_closed=14000):
        self.nav = nav
        self.s = s
        self.g = g
        self.w = weight
        self.max_closed = max_closed
        self.gs = {s: 0.0}
        self.came = {}
        self.closed = set()
        self.gj, self.gi = divmod(g, nav.nx)
        self.open = [(self._h(s), 0.0, s)]
        self.best = s
        self.besth = self._h(s)
        self.done = False
        self.path = None

    def _h(self, idx):
        j, i = divmod(idx, self.nav.nx)
        dx, dy = abs(i - self.gi), abs(j - self.gj)
        return (dx + dy) - 0.5858 * min(dx, dy)

    def _build(self, end):
        out = [end]
        while out[-1] in self.came:
            out.append(self.came[out[-1]])
        out.reverse()
        self.path = [self.nav.center(k) for k in out]
        self.done = True

    def step(self, budget):
        if self.done:
            return True
        nav = self.nav
        nx, ny = nav.nx, nav.ny
        block = nav.block
        t_end = time.perf_counter() + budget
        n = 0
        openh = self.open
        gs, came, closed = self.gs, self.came, self.closed
        while openh:
            n += 1
            if (n & 63) == 0 and time.perf_counter() > t_end:
                return False
            f, gc, cur = heapq.heappop(openh)
            if cur in closed:
                continue
            if cur == self.g:
                self._build(cur)
                return True
            closed.add(cur)
            if len(closed) > self.max_closed:
                break
            hcur = f - self.w * 0  # unused
            cj, ci = divmod(cur, nx)
            for di, dj, c in _NB:
                i, j = ci + di, cj + dj
                if i < 0 or j < 0 or i >= nx or j >= ny:
                    continue
                k = j * nx + i
                if block[k] or k in closed:
                    continue
                if di and dj and (block[cj * nx + i] or block[j * nx + ci]):
                    continue
                ng = gc + c
                if ng < gs.get(k, 1e18):
                    gs[k] = ng
                    came[k] = cur
                    h = self._h(k)
                    if h < self.besth:
                        self.besth = h
                        self.best = k
                    heapq.heappush(openh, (ng + self.w * h, ng, k))
        self._build(self.best)
        return True