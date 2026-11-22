"""Mech Royale bot: stay inside the zone, avoid needless fights, shoot what is in range, farm cubes.

Standard library only. Navigation uses a coarse walkability grid built from the map at start-up and a
breadth-first distance field computed in a background thread, so a decision never waits for it.
"""
import json
import math
import os
import random
import sys
import threading
import time
from array import array
from collections import deque

import mechfight as mf

CELL = 6.0  # navigation cell, meters
FINE = 2.0  # rasterization cell, meters
NB = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]

UPGRADE_ORDER = ["armor", "plating", "nanites", "gun_damage", "gun_rate", "gun_mag", "servos", "gyros",
                 "laser_damage", "sensors", "reactor", "capacitor", "gun_velocity", "jets"]


def wrap(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


class Terrain:
    def __init__(self, mapdir, files):
        self.world = json.load(open(os.path.join(mapdir, "v2", "world.json")))
        t = self.world["terrain"]
        self.nx, self.ny, self.minx, self.miny, self.step = t["nx"], t["ny"], t["minX"], t["minY"], t["step"]
        raw = array("h")
        with open(os.path.join(mapdir, "v2", t["file"]), "rb") as f:
            raw.frombytes(f.read())
        if sys.byteorder == "big":
            raw.byteswap()
        self.h = raw

    def z(self, x, y):
        fx = (x - self.minx) / self.step
        fy = (y - self.miny) / self.step
        i = int(max(0, min(self.nx - 2, math.floor(fx))))
        j = int(max(0, min(self.ny - 2, math.floor(fy))))
        tx = min(1.0, max(0.0, fx - i))
        ty = min(1.0, max(0.0, fy - j))
        h, nx = self.h, self.nx
        a = h[j * nx + i]
        b = h[j * nx + i + 1]
        c = h[(j + 1) * nx + i]
        d = h[(j + 1) * nx + i + 1]
        return 0.01 * ((a * (1 - tx) + b * tx) * (1 - ty) + (c * (1 - tx) + d * tx) * ty)


def inside(poly, x, y):
    n = len(poly) // 2
    c = False
    j = n - 1
    for i in range(n):
        xi, yi, xj, yj = poly[2 * i], poly[2 * i + 1], poly[2 * j], poly[2 * j + 1]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            c = not c
        j = i
    return c


class Nav:
    """Walkability grid plus a background distance-field solver."""

    def __init__(self, terrain):
        w = terrain.world
        b = w["bounds"]
        self.x0, self.y0 = b["minX"] + 3, b["minY"] + 3
        self.nx = int((b["maxX"] - 3 - self.x0) / CELL)
        self.ny = int((b["maxY"] - 3 - self.y0) / CELL)
        fnx = int((b["maxX"] - b["minX"]) / FINE) + 2
        fny = int((b["maxY"] - b["minY"]) / FINE) + 2
        fx0, fy0 = b["minX"], b["minY"]
        fine = bytearray(fnx * fny)
        for p in w["prisms"]:
            if p["kind"] == "bridge":
                continue
            ring = p["ring"]
            xs, ys = ring[0::2], ring[1::2]
            holes = p.get("holes") or []
            hole_list = holes if (holes and isinstance(holes[0], list)) else ([holes] if holes else [])
            i0 = max(0, int((min(xs) - fx0) / FINE) - 1)
            i1 = min(fnx - 1, int((max(xs) - fx0) / FINE) + 1)
            j0 = max(0, int((min(ys) - fy0) / FINE) - 1)
            j1 = min(fny - 1, int((max(ys) - fy0) / FINE) + 1)
            base, top = p["base"], p["top"]
            for j in range(j0, j1 + 1):
                y = fy0 + (j + 0.5) * FINE
                for i in range(i0, i1 + 1):
                    x = fx0 + (i + 0.5) * FINE
                    if not inside(ring, x, y):
                        continue
                    if any(inside(h, x, y) for h in hole_list):
                        continue
                    g = terrain.z(x, y)
                    if base < g + 9.0 and top > g + 1.2:
                        fine[j * fnx + i] = 1
        # steep ground
        slope_lim = math.tan(math.radians(48)) * FINE
        for j in range(1, fny - 1):
            y = fy0 + (j + 0.5) * FINE
            for i in range(1, fnx - 1):
                if fine[j * fnx + i]:
                    continue
                x = fx0 + (i + 0.5) * FINE
                g = terrain.z(x, y)
                if abs(terrain.z(x + FINE, y) - g) > slope_lim or abs(terrain.z(x, y + FINE) - g) > slope_lim:
                    fine[j * fnx + i] = 1
        # loose grid: only the raw cell under the centre counts (used when the strict grid has no way through)
        self.free_loose = bytearray(self.nx * self.ny)
        for cj in range(self.ny):
            for ci in range(self.nx):
                i = int((self.x0 + (ci + 0.5) * CELL - fx0) / FINE)
                j = int((self.y0 + (cj + 0.5) * CELL - fy0) / FINE)
                self.free_loose[cj * self.nx + ci] = 0 if fine[j * fnx + i] else 1
        # dilate by one fine cell
        dil = bytearray(fine)
        for j in range(1, fny - 1):
            row = j * fnx
            for i in range(1, fnx - 1):
                if fine[row + i]:
                    for di in (-1, 0, 1):
                        for dj in (-1, 0, 1):
                            dil[row + di + dj * fnx + i] = 1
        # coarse walkability: blocked if any of 3x3 samples is blocked
        self.free = bytearray(self.nx * self.ny)
        for cj in range(self.ny):
            for ci in range(self.nx):
                cx = self.x0 + (ci + 0.5) * CELL
                cy = self.y0 + (cj + 0.5) * CELL
                ok = 1
                for dx in (-FINE, 0.0, FINE):
                    for dy in (-FINE, 0.0, FINE):
                        i = int((cx + dx - fx0) / FINE)
                        j = int((cy + dy - fy0) / FINE)
                        if dil[j * fnx + i]:
                            ok = 0
                self.free[cj * self.nx + ci] = ok
        self.field = None  # (goal_cell, dist list)
        self.want = None
        self.lock = threading.Lock()
        self.cv = threading.Condition(self.lock)
        threading.Thread(target=self._worker, daemon=True).start()

    def cell(self, x, y):
        return (int((x - self.x0) / CELL), int((y - self.y0) / CELL))

    def center(self, c):
        return (self.x0 + (c[0] + 0.5) * CELL, self.y0 + (c[1] + 0.5) * CELL)

    def ok(self, c):
        return 0 <= c[0] < self.nx and 0 <= c[1] < self.ny and self.free[c[1] * self.nx + c[0]]

    def nearest_free(self, c, maxr=12):
        c = (max(0, min(self.nx - 1, c[0])), max(0, min(self.ny - 1, c[1])))
        if self.ok(c):
            return c
        for r in range(1, maxr):
            for dx in range(-r, r + 1):
                for dy in (-r, r):
                    for cc in ((c[0] + dx, c[1] + dy), (c[0] + dy, c[1] + dx)):
                        if self.ok(cc):
                            return cc
        return None

    def nearest_loose(self, c, maxr=8):
        for r in range(0, maxr):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    cc = (c[0] + dx, c[1] + dy)
                    if 0 <= cc[0] < self.nx and 0 <= cc[1] < self.ny and self.free_loose[cc[1] * self.nx + cc[0]]:
                        return cc
        return None

    def request(self, x, y):
        g = self.nearest_free(self.cell(x, y))
        if g is None:
            return
        with self.cv:
            cur = self.field[0] if self.field else None
            if g != cur and g != self.want:
                self.want = g
                self.cv.notify()

    def _worker(self):
        while True:
            with self.cv:
                while self.want is None:
                    self.cv.wait()
                g = self.want
                self.want = None
            try:
                dist = self._bfs(g, self.free)
                gl = g if self.free_loose[g[1] * self.nx + g[0]] else (self.nearest_loose(g) or g)
                dloose = self._bfs(gl, self.free_loose)
            except Exception:
                continue
            with self.lock:
                self.field = (g, dist, dloose)

    def _bfs(self, g, free):
        nx, ny = self.nx, self.ny
        dist = [-1] * (nx * ny)
        dist[g[1] * nx + g[0]] = 0
        dq = deque([g])
        while dq:
            cx, cy = dq.popleft()
            d = dist[cy * nx + cx] + 1
            for dx, dy in NB:
                x, y = cx + dx, cy + dy
                if 0 <= x < nx and 0 <= y < ny:
                    k = y * nx + x
                    if dist[k] < 0 and free[k]:
                        if dx and dy and not (free[cy * nx + x] and free[y * nx + cx]):
                            continue
                        dist[k] = d
                        dq.append((x, y))
        return dist

    def direction(self, x, y, gx, gy):
        """Unit vector toward (gx, gy) along the path, or straight if there is no field yet."""
        straight = (gx - x, gy - y)
        n = math.hypot(*straight) or 1.0
        straight = (straight[0] / n, straight[1] / n)
        with self.lock:
            f = self.field
        if f is None:
            return straight, False
        g, dist, dloose = f
        if math.hypot(self.center(g)[0] - gx, self.center(g)[1] - gy) > 3 * CELL:
            return straight, False
        c = self.cell(x, y)
        if not (0 <= c[0] < self.nx and 0 <= c[1] < self.ny):
            return straight, False
        d0 = dist[c[1] * self.nx + c[0]]
        if d0 < 0 and dloose[c[1] * self.nx + c[0]] >= 0:
            dist = dloose
            d0 = dist[c[1] * self.nx + c[0]]
        if d0 < 0:
            # standing on a blocked cell: pick the best free neighbour
            best = None
            for dx, dy in NB:
                cc = (c[0] + dx, c[1] + dy)
                if 0 <= cc[0] < self.nx and 0 <= cc[1] < self.ny:
                    dd = dist[cc[1] * self.nx + cc[0]]
                    if dd >= 0 and (best is None or dd < best[0]):
                        best = (dd, cc)
            if best is None:
                return straight, False
            c, d0 = best[1], best[0]
        if d0 <= 1:
            return straight, True
        cur = c
        for _ in range(4):
            best = None
            for dx, dy in NB:
                cc = (cur[0] + dx, cur[1] + dy)
                if 0 <= cc[0] < self.nx and 0 <= cc[1] < self.ny:
                    dd = dist[cc[1] * self.nx + cc[0]]
                    if dd >= 0 and dd < dist[cur[1] * self.nx + cur[0]] and (best is None or dd < best[0]):
                        best = (dd, cc)
            if best is None:
                break
            cur = best[1]
        px, py = self.center(cur)
        v = (px - x, py - y)
        n = math.hypot(*v)
        if n < 1e-6:
            return straight, True
        return (v[0] / n, v[1] / n), True


class Bot:
    name = "survivor-py"

    def __init__(self, hello):
        self.cfg = hello["config"]
        self.slot = hello["slot"]
        self.rng = random.Random(hello.get("seed", 0))
        t0 = time.time()
        self.nav = None
        try:
            mapdir = hello["map"]["dir"]
            self.nav = Nav(Terrain(mapdir, hello["map"].get("files")))
        except Exception:
            mf.log("nav init failed:\n" + __import__("traceback").format_exc())
        mf.log("init %.2fs" % (time.time() - t0))
        self.strafe_dir = 1
        self.strafe_until = 0.0
        self.last_pos = None
        self.stuck_ticks = 0
        self.unstick_until = 0.0
        self.unstick_dir = [1.0, 0.0]
        self.flee_until = 0.0
        self.flee_bearing = None
        self.goal = None
        self.goal_key = None
        self.last_hit_t = -99.0

    # ---------------------------------------------------------------- helpers
    def pick_upgrade(self, offer):
        best, bi = 99, 0
        for i, ch in enumerate(offer["choices"]):
            cid = ch["id"]
            r = UPGRADE_ORDER.index(cid) if cid in UPGRADE_ORDER else 50
            if r < best:
                best, bi = r, i
        return bi

    def zone_goal(self, obs, x, y):
        z = obs["zone"]
        nc, nr = z["next_center"], z["next_radius"]
        key = (round(nc[0]), round(nc[1]), round(nr))
        if key != self.goal_key:
            self.goal_key = key
            # stand inside the next circle, on the side we are already on, not in the middle
            dx, dy = x - nc[0], y - nc[1]
            d = math.hypot(dx, dy)
            r = nr * 0.55
            if d > r:
                self.goal = (nc[0] + dx / d * r, nc[1] + dy / d * r)
            else:
                self.goal = (x, y)
        return self.goal

    def lead(self, me, tg, speed):
        px, py, pz = tg["pos"]
        vx, vy, vz = tg.get("vel", [0, 0, 0])
        mx, my, mz = me["pos"]
        t = 0.0
        for _ in range(3):
            d = math.dist((px + vx * t, py + vy * t), (mx, my))
            t = d / speed
        tx, ty = px + vx * t, py + vy * t
        tz = pz + 4.5 + vz * t * 0.5
        d = math.hypot(tx - mx, ty - my)
        return math.atan2(ty - my, tx - mx), math.atan2(tz - (mz + 7.0), max(d, 1e-3)), d

    # ---------------------------------------------------------------- main
    def act(self, obs):
        me = obs["self"]
        x, y, z = me["pos"]
        t = obs["t"]
        act = {"tick": obs["tick"]}
        if obs.get("offer"):
            act["pick"] = self.pick_upgrade(obs["offer"])

        weapons = me["weapons"]
        gun = weapons[0]
        laser_i = next((i for i, w in enumerate(weapons) if w and w["kind"] == "laser"), None)

        # --- loot: take a laser when we are standing next to one
        reach = self.cfg["loot"]["pickup_radius"]
        pk_target = None
        for p in obs["pickups"]:
            d = math.dist(p["pos"][:2], (x, y))
            if p["kind"] == "laser" and laser_i is None and (pk_target is None or d < pk_target[0]):
                pk_target = (d, p)
        if pk_target and pk_target[0] < reach - 0.5 and abs(pk_target[1]["pos"][2] - z) < 3.5:
            if weapons[1] is None:
                act["take"] = pk_target[1]["id"]
            elif laser_i is None and weapons[1]["kind"] in ("mortar", "lightning"):
                act["take"] = pk_target[1]["id"]
                act["drop"] = 1

        mechs = [m for m in obs["mechs"] if m["kind"] == "mech"]
        npcs = [m for m in obs["mechs"] if m["kind"] == "npc"]
        mechs.sort(key=lambda m: math.dist(m["pos"][:2], (x, y)))
        npcs.sort(key=lambda m: math.dist(m["pos"][:2], (x, y)))
        enemy = mechs[0] if mechs else None
        edist = math.dist(enemy["pos"][:2], (x, y)) if enemy else 1e9

        # --- damage memory (attacker we cannot see)
        for ev in obs["events"]:
            if ev.get("type") == "damaged" and ev.get("weapon") != "zone":
                self.last_hit_t = t
                if ev.get("bearing") is not None:
                    self.flee_bearing = ev["bearing"]
                    self.flee_until = t + 1.2

        hp_frac = me["hp"] / max(1.0, me["max_hp"])
        zone = obs["zone"]
        zc = zone["center"]
        zr = zone["radius"]
        zd = math.dist(zc, (x, y))
        margin = zr - zd
        zgoal = self.zone_goal(obs, x, y)

        # --- where do we want to go?
        goal = zgoal
        reason = "zone"
        want_boost = False

        # cubes and drones when it is calm
        calm = enemy is None and t - self.last_hit_t > 3.0
        if calm and margin > 40:
            best = None
            for c in obs["cubes"]:
                d = math.dist(c["pos"][:2], (x, y))
                if d < 70 and (best is None or d < best[0]):
                    best = (d, c["pos"])
            if best:
                goal, reason = (best[1][0], best[1][1]), "cube"
            elif pk_target and pk_target[0] < 60:
                goal, reason = (pk_target[1]["pos"][0], pk_target[1]["pos"][1]), "laser"
            elif npcs and hp_frac > 0.5 and math.dist(npcs[0]["pos"][:2], (x, y)) < 90:
                goal, reason = (npcs[0]["pos"][0], npcs[0]["pos"][1]), "npc"

        # supply drop (full heal + cubes) when it is close, calm and inside the safe area
        for d in obs.get("drops", []):
            dd = math.dist(d["pos"][:2], (x, y))
            if dd < 110 and enemy is None and d["lands_in"] < 14 and math.dist(d["pos"][:2], zone["next_center"]) < zone["next_radius"] - 15:
                goal, reason = (d["pos"][0], d["pos"][1]), "drop"
                break

        # outside or near the edge of the zone: go back in
        if margin < 12:
            goal, reason = zgoal if margin > -1e9 else zc, "zone!"
            if margin < 0:
                goal = zc
            want_boost = me["energy"] > 20

        # --- navigation
        nav = self.nav
        if nav is not None:
            nav.request(goal[0], goal[1])
            dirv, ok = nav.direction(x, y, goal[0], goal[1])
        else:
            n = math.hypot(goal[0] - x, goal[1] - y) or 1.0
            dirv = ((goal[0] - x) / n, (goal[1] - y) / n)
        gd = math.hypot(goal[0] - x, goal[1] - y)
        move = [dirv[0], dirv[1]]
        if gd < 6 and reason in ("zone", "zone!"):
            move = [0.0, 0.0]  # arrived: stand still and regenerate

        # --- fighting
        fire = [False, False]
        aim = list(me["aim"])
        target = None
        if enemy is not None and edist < 230:
            target = enemy
        elif npcs and reason == "npc" and math.dist(npcs[0]["pos"][:2], (x, y)) < 150:
            target = npcs[0]
        elif npcs and t - self.last_hit_t < 2 and math.dist(npcs[0]["pos"][:2], (x, y)) < 60:
            target = npcs[0]

        if target is not None:
            gspeed = (gun["stats"]["speed"] if gun else 252.0)
            yaw, pitch, d = self.lead(me, target, gspeed)
            aim = [yaw, max(-0.7, min(1.3, pitch))]
            err = abs(wrap(yaw - me["aim"][0]))
            if gun and d < min(180.0, gun["stats"]["range"] * 0.95) and err < 0.12:
                fire[0] = True
            # laser: charge when lined up, release at full charge
            if laser_i is not None:
                lw = weapons[laser_i]
                ch = lw.get("charge", 0) or 0
                ly = math.atan2(target["pos"][1] - y, target["pos"][0] - x)
                lp = math.atan2(target["pos"][2] + 4.5 - (z + 7.0), max(d, 1e-3))
                lerr = abs(wrap(ly - me["aim"][0]))
                if ch > 0:
                    aim = [ly, max(-0.7, min(1.3, lp))]
                    fire[laser_i] = not (ch >= 0.7 or lerr > 0.15)
                elif lw["cooldown"] <= 0 and lerr < 0.1 and d < 240 and target["kind"] == "mech":
                    fire[laser_i] = True
            # movement while fighting: keep distance, strafe, do not walk into the open
            if target["kind"] == "mech":
                if t > self.strafe_until:
                    self.strafe_dir = self.rng.choice((-1, 1))
                    self.strafe_until = t + self.rng.uniform(0.8, 2.2)
                ex, ey = target["pos"][0] - x, target["pos"][1] - y
                en = math.hypot(ex, ey) or 1.0
                ux, uy = ex / en, ey / en
                sx, sy = -uy * self.strafe_dir, ux * self.strafe_dir
                radial = 0.0
                if en < 50:
                    radial = -0.8
                elif en > 140:
                    radial = 0.2
                if hp_frac < 0.35:
                    radial = -1.0
                    want_boost = me["energy"] > 25
                fx = sx * 0.8 + ux * radial
                fy = sy * 0.8 + uy * radial
                # stay roughly in the zone: blend with the way to the goal when near the edge
                if margin < 40:
                    fx += dirv[0] * 0.8
                    fy += dirv[1] * 0.8
                n = math.hypot(fx, fy) or 1.0
                if margin >= 12 or reason != "zone!":
                    move = [fx / n, fy / n]
            else:
                # drone: stay at a comfortable range
                if d < 30:
                    move = [0.0, 0.0] if move == [0.0, 0.0] else move
        elif t < self.flee_until and self.flee_bearing is not None:
            # hit by someone we cannot see: face them and sidestep
            aim = [self.flee_bearing, 0.0]
            sx, sy = -math.sin(self.flee_bearing), math.cos(self.flee_bearing)
            move = [sx * self.strafe_dir * 0.8 + dirv[0] * 0.5, sy * self.strafe_dir * 0.8 + dirv[1] * 0.5]
            n = math.hypot(*move) or 1.0
            move = [move[0] / n, move[1] / n]
        else:
            if math.hypot(*move) > 0.1:
                aim = [math.atan2(move[1], move[0]), 0.0]
            else:
                aim = list(me["aim"])

        # --- reloading
        reload = False
        if gun:
            ammo = gun.get("ammo", 99)
            if ammo <= 0 and gun.get("reload", 0) <= 0:
                reload = True
            elif target is None and ammo < gun["stats"]["mag"] and gun.get("reload", 0) <= 0:
                reload = True

        # --- stuck detection
        jump = False
        if self.last_pos is not None:
            moved = math.dist((x, y), self.last_pos)
            if math.hypot(*move) > 0.5 and moved < 0.6 and me["grounded"]:
                self.stuck_ticks += 1
            else:
                self.stuck_ticks = max(0, self.stuck_ticks - 1)
        self.last_pos = (x, y)
        if self.stuck_ticks >= 8 and t > self.unstick_until:
            a = self.rng.uniform(0, 2 * math.pi)
            self.unstick_dir = [math.cos(a), math.sin(a)]
            self.unstick_until = t + 1.2
            self.stuck_ticks = 0
        if t < self.unstick_until:
            move = list(self.unstick_dir)
            jump = me["grounded"] and me["energy"] > 40
            want_boost = me["energy"] > 30

        act["move"] = [round(move[0], 3), round(move[1], 3)]
        act["aim"] = [round(aim[0], 4), round(aim[1], 4)]
        act["fire"] = fire
        act["boost"] = bool(want_boost and (math.hypot(*move) > 0.1))
        act["jump"] = bool(jump)
        if reload:
            act["reload"] = True
        return act


class SafeBot:
    """Wraps Bot so any failure still returns a sane action instead of coasting blind."""

    name = Bot.name

    def __init__(self, hello):
        self.bot = Bot(hello)

    def act(self, obs):
        try:
            return self.bot.act(obs)
        except Exception:
            mf.log("act error:\n" + __import__("traceback").format_exc())
            me = obs["self"]
            zc = obs["zone"]["next_center"]
            yaw = math.atan2(zc[1] - me["pos"][1], zc[0] - me["pos"][0])
            return {"tick": obs["tick"], "move": [math.cos(yaw), math.sin(yaw)], "aim": [yaw, 0.0]}


if __name__ == "__main__":
    sys.setswitchinterval(0.001)
    mf.run(SafeBot)