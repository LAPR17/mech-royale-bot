#!/usr/bin/env python3
"""Mech Royale bot: placement-first. Stays in the zone, loots, fights with lead aiming, strafes,
dodges shells and lasers, kites when hurt."""
import math
import os
import random
import time
import traceback

import mechfight as mf

try:
    from nav import Nav, Search
except Exception:  # pragma: no cover
    Nav = None
    Search = None

TAU = math.tau


def wrap(a):
    while a > math.pi:
        a -= TAU
    while a < -math.pi:
        a += TAU
    return a


def hyp(a, b):
    return math.sqrt(a * a + b * b)


def unit(x, y):
    d = hyp(x, y)
    if d < 1e-9:
        return 0.0, 0.0
    return x / d, y / d


UPG = {
    "armor": 9.0, "plating": 8.6, "nanites": 8.0, "gun_damage": 8.2, "gun_rate": 7.8,
    "lightning_efficiency": 7.2, "lightning_range": 6.8, "lightning_chain": 6.4,
    "laser_damage": 7.0, "laser_charge": 6.6, "gyros": 6.2, "servos": 5.8, "sensors": 6.0,
    "gun_velocity": 5.6, "gun_mag": 5.2, "capacitor": 5.0, "reactor": 4.8, "jets": 3.5,
    "laser_pierce": 3.0, "mortar_reload": 3.2, "mortar_splash": 3.0, "mortar_velocity": 3.0,
}
WRANK = {"lightning": 3, "laser": 2.5, "mortar": 1}


class Bot:
    name = "apex-mech"

    def __init__(self, hello):
        self.cfg = hello["config"]
        self.rng = random.Random(hello.get("seed", 1))
        self.nav = None
        try:
            p = os.path.join(hello["map"]["dir"], "v2", "world.json")
            if Nav is not None and os.path.exists(p):
                self.nav = Nav(p)
        except Exception:
            mf.log("nav load failed", traceback.format_exc())
        self.items = {}
        self.search = None
        self.path = None
        self.path_goal = -2
        self.path_idx = 0
        self.path_t = -99.0
        self.strafe = 1
        self.strafe_until = 0.0
        self.target_id = None
        self.dmg_t = -99.0
        self.dmg_bearing = None
        self.hist = []
        self.unstick_until = 0.0
        self.unstick_dir = (1.0, 0.0)
        self.scan = 0.0
        self.t = 0.0
        self.t0 = 0.0

    # ------------------------------------------------------------------ small helpers
    def los(self, me, tp):
        if self.nav is None:
            return True
        try:
            return self.nav.los(me["pos"][0], me["pos"][1], tp[0], tp[1], me["pos"][2])
        except Exception:
            return True

    def choose_upgrade(self, obs):
        best, bi = -1.0, 0
        for i, c in enumerate(obs["offer"]["choices"]):
            s = UPG.get(c["id"], 4.0)
            if s > best:
                best, bi = s, i
        return bi

    def region(self, obs):
        z = obs["zone"]
        t = obs["t"]
        c, r = z["center"], z["radius"]
        if z["shrink_start"] - 12 <= t < z["shrink_end"] + 1:
            c, r = z["next_center"], z["next_radius"]
        return c, max(r, 6.0)

    def update_items(self, obs, me):
        t = obs["t"]
        x, y = me["pos"][0], me["pos"][1]
        for p in obs["pickups"]:
            self.items[p["id"]] = ("pickup", p["kind"], p["pos"], t)
        for c in obs["cubes"]:
            self.items[c["id"]] = ("cube", "cube", c["pos"], t)
        for c in obs["crates"]:
            self.items[c["id"]] = ("crate", "crate", c["pos"], t)
        dead = []
        for k, (kind, sub, pos, seen) in self.items.items():
            if t - seen > 90:
                dead.append(k)
            elif seen < t and hyp(pos[0] - x, pos[1] - y) < 18:
                dead.append(k)
        for k in dead:
            self.items.pop(k, None)

    # ------------------------------------------------------------------ navigation
    def nav_to(self, me, gx, gy):
        """Unit vector toward (gx,gy) following a path; None if already there."""
        x, y = me["pos"][0], me["pos"][1]
        d = hyp(gx - x, gy - y)
        if d < 2.5:
            return None
        if self.nav is None or Search is None:
            return unit(gx - x, gy - y)
        nav = self.nav
        if d < 30 and nav.clear_seg(x, y, gx, gy):
            return unit(gx - x, gy - y)
        gcell = nav.cell(gx, gy)
        gcell = nav.nearest_free(gcell, 8)
        scell = nav.nearest_free(nav.cell(x, y), 4)
        if gcell < 0 or scell < 0:
            return unit(gx - x, gy - y)
        need = (gcell != self.path_goal) or (self.t - self.path_t > 2.0) or self.path is None
        if need and (self.search is None or self.search.g != gcell):
            if time.perf_counter() - self.t0 < 0.02:
                self.search = Search(nav, scell, gcell)
        if self.search is not None:
            left = 0.034 - (time.perf_counter() - self.t0)
            if left > 0.003 and self.search.step(min(left, 0.012)):
                self.path = self.search.path
                self.path_goal = self.search.g
                self.path_idx = 0
                self.path_t = self.t
                self.search = None
        if not self.path:
            return unit(gx - x, gy - y)
        # advance along the path
        pth = self.path
        bi, bd = self.path_idx, 1e18
        for k in range(self.path_idx, min(len(pth), self.path_idx + 40)):
            dd = hyp(pth[k][0] - x, pth[k][1] - y)
            if dd < bd:
                bd, bi = dd, k
        self.path_idx = bi
        if bd > 25:  # far off the path, drop it
            self.path = None
            self.path_goal = -2
            return unit(gx - x, gy - y)
        tgt = pth[min(bi + 1, len(pth) - 1)]
        for k in range(bi + 1, min(len(pth), bi + 14)):
            wx, wy = pth[k]
            if hyp(wx - x, wy - y) > 34:
                break
            if nav.clear_seg(x, y, wx, wy):
                tgt = pth[k]
        if hyp(tgt[0] - x, tgt[1] - y) < 1.5 and bi + 2 < len(pth):
            tgt = pth[bi + 2]
        return unit(tgt[0] - x, tgt[1] - y)

    def safe_dir(self, me, vx, vy, look=9.0):
        if self.nav is None or (abs(vx) < 1e-6 and abs(vy) < 1e-6):
            return vx, vy
        x, y = me["pos"][0], me["pos"][1]
        ux, uy = unit(vx, vy)
        if self.nav.clear_seg(x, y, x + ux * look, y + uy * look):
            return vx, vy
        base = math.atan2(uy, ux)
        mag = hyp(vx, vy)
        for off in (0.6, -0.6, 1.2, -1.2, 1.9, -1.9, 2.6, -2.6, 3.14):
            a = base + off
            cx, cy = math.cos(a), math.sin(a)
            if self.nav.clear_seg(x, y, x + cx * look, y + cy * look):
                return cx * mag, cy * mag
        return vx, vy

    # ------------------------------------------------------------------ aiming
    def lead_aim(self, me, tp, tv, speed):
        sx, sy = me["pos"][0], me["pos"][1]
        mz = me["pos"][2] + me["stats"].get("muzzle_height", 7.0)
        t = 0.0
        px, py, pz = tp
        dd = 1.0
        for _ in range(4):
            px, py, pz = tp[0] + tv[0] * t, tp[1] + tv[1] * t, tp[2] + tv[2] * t
            dd = math.sqrt((px - sx) ** 2 + (py - sy) ** 2 + (pz - mz) ** 2)
            t = min(dd / max(speed, 1.0), 1.1)
        yaw = math.atan2(py - sy, px - sx)
        pitch = math.atan2(pz - mz, hyp(px - sx, py - sy))
        st = me["stats"]
        pitch = max(st.get("min_pitch", -0.7), min(st.get("max_pitch", 1.3), pitch))
        return yaw, pitch, dd

    def mortar_aim(self, me, tpos, tvel, speed, g):
        sx, sy = me["pos"][0], me["pos"][1]
        sz = me["pos"][2] + me["stats"].get("height", 9.0) + 0.5
        tf = 1.5
        for _ in range(4):
            px, py = tpos[0] + tvel[0] * tf, tpos[1] + tvel[1] * tf
            D = hyp(px - sx, py - sy)
            dz = (tpos[2] + 1.5) - sz
            if D < 5:
                return None
            sol = None
            prev_f = None
            th = -0.6
            while th <= 1.2:
                c = math.cos(th)
                f = D * math.tan(th) - g * D * D / (2 * speed * speed * c * c) - dz
                if prev_f is not None and prev_f < 0 <= f:
                    sol = th
                    break
                prev_f = f
                th += 0.01
            if sol is None:
                return None
            tf = D / (speed * math.cos(sol))
            if tf > 6.8:
                return None
        yaw = math.atan2(py - sy, px - sx)
        return yaw, sol, D, tf

    @staticmethod
    def aim_err(me, yaw, pitch):
        ay, ap = me["aim"]
        ey = wrap(yaw - ay) * math.cos(pitch)
        return hyp(ey, pitch - ap)

    # ------------------------------------------------------------------ main
    def act(self, obs):
        self.t0 = time.perf_counter()
        t = obs["t"]
        self.t = t
        me = obs["self"]
        x, y, z = me["pos"]
        st = me["stats"]
        hpf = me["hp"] / max(me["max_hp"], 1.0)
        a = {"tick": obs["tick"]}

        for e in obs["events"]:
            if e["type"] == "damaged" and e.get("bearing") is not None and e.get("weapon") != "zone":
                self.dmg_t = t
                self.dmg_bearing = e["bearing"]
            elif e["type"] == "took":
                self.items.pop(e["id"], None)
        self.update_items(obs, me)
        if obs.get("offer"):
            a["pick"] = self.choose_upgrade(obs)

        rc, rr = self.region(obs)
        dzone = hyp(x - rc[0], y - rc[1])
        zone_urgent = (not me["in_zone"]) or dzone > rr - 10
        soft_out = dzone > max(rr * 0.8 - 8, 8)

        w0 = me["weapons"][0]
        w1 = me["weapons"][1] if len(me["weapons"]) > 1 else None
        kind1 = w1["kind"] if w1 else None

        enemies = [m for m in obs["mechs"] if m["kind"] == "mech" and m["id"] != me["id"]]
        npcs = [m for m in obs["mechs"] if m["kind"] == "npc"]

        # ---------------- pick the combat target
        target = None
        best = 1e18
        for m in enemies:
            d = hyp(m["pos"][0] - x, m["pos"][1] - y)
            if d > 235:
                continue
            los = self.los(me, m["pos"])
            hp = m.get("hp", 1000.0) / max(m.get("max_hp", 1400.0), 1.0)
            s = d + (0 if los else 45) + hp * 25
            if m["id"] == self.target_id:
                s -= 18
            if s < best:
                best, target = s, m
        self.target_id = target["id"] if target else None

        mv = None  # desired move (x,y) or None for stand still
        boost = False
        jump = False
        aim = None
        fire0 = False
        fire1 = False
        reload_now = False
        tdist = None
        label = ""

        # ---------------- dodge: incoming mortar shells and gun bullets
        dodge = None
        for pr in obs["projectiles"]:
            try:
                px, py, pz = pr["pos"]
                vx, vy, vz = pr["vel"]
                if pr["kind"] == "mortar":
                    g = self.cfg["weapons"]["mortar"].get("gravity", 22.0)
                    tt = 0.0
                    ix, iy = px, py
                    while tt < 7.0:
                        tt += 0.05
                        ix, iy = px + vx * tt, py + vy * tt
                        iz = pz + vz * tt - 0.5 * g * tt * tt
                        if iz <= z + 1.0:
                            break
                    if tt < 3.0 and hyp(ix - x, iy - y) < 11:
                        ux, uy = unit(x - ix, y - iy)
                        if ux == 0 and uy == 0:
                            ux, uy = 1.0, 0.0
                        dodge = (ux, uy, tt < 1.6)
                else:
                    rx, ry = x - px, y - py
                    sp2 = vx * vx + vy * vy
                    if sp2 < 1.0:
                        continue
                    tc = (rx * vx + ry * vy) / sp2
                    if 0.0 < tc < 0.42:
                        cx, cy = px + vx * tc, py + vy * tc
                        miss = hyp(cx - x, cy - y)
                        if miss < 3.6 and abs(pz + vz * tc - (z + 4.5)) < 7:
                            sx_, sy_ = -vy, vx
                            side = 1.0 if (sx_ * (x - cx) + sy_ * (y - cy)) >= 0 else -1.0
                            ux, uy = unit(sx_ * side, sy_ * side)
                            if dodge is None:
                                dodge = (ux, uy, False)
            except Exception:
                continue

        # laser telegraph dodge
        for m in enemies:
            if m.get("charging"):
                mx, my = m["pos"][0], m["pos"][1]
                d = hyp(x - mx, y - my)
                if d < 300:
                    ang = wrap(math.atan2(y - my, x - mx) - m["aim"][0])
                    if abs(ang) < 0.45 and d * math.sin(abs(ang)) < 7.5:
                        px_, py_ = -math.sin(m["aim"][0]), math.cos(m["aim"][0])
                        side = 1.0 if (px_ * (x - mx) + py_ * (y - my)) >= 0 else -1.0
                        dodge = (px_ * side, py_ * side, me["energy"] > 25)

        # ---------------- combat
        if target is not None:
            tp = target["pos"]
            tv = target.get("vel", [0, 0, 0])
            d = hyp(tp[0] - x, tp[1] - y)
            tdist = d
            los = self.los(me, tp)
            tw = target.get("weapons", [])
            ccz = tp[2] + 4.5 * (1.0 + 0.05 * target.get("cubes", 0))
            center = (tp[0], tp[1], ccz)

            if kind1 == "lightning":
                lo, hi = 14.0, 30.0
            elif kind1 == "laser":
                lo, hi = 55.0, 120.0
            else:
                lo, hi = 32.0, 66.0
            if "lightning" in tw and kind1 != "lightning":
                lo, hi = max(lo, 60.0), max(hi, 110.0)
            if "mortar" in tw and kind1 != "mortar":
                lo, hi = min(lo, 30.0), min(hi, 60.0)
            if hpf < 0.35:
                lo, hi = lo + 25, hi + 40

            gspeed = w0["stats"]["speed"] if w0 and "stats" in w0 else 252.0
            yaw, pitch, dd = self.lead_aim(me, center, tv, gspeed)
            aim = [yaw, pitch]
            err = self.aim_err(me, yaw, pitch)
            gun_ok = w0 is not None and w0.get("ammo", 1) > 0 and w0.get("reload", 0) <= 0
            gun_range = w0["stats"].get("range", 230) if w0 and "stats" in w0 else 230
            if gun_ok and los and d < gun_range - 5 and err * dd < 3.3:
                fire0 = True

            used_mortar = False
            if w1 is not None and w1.get("cooldown", 0) <= 0.4:
                s1 = w1.get("stats", {})
                if kind1 == "lightning":
                    rng_l = s1.get("range", 45.0)
                    cone = math.radians(s1.get("cone_deg", 30.0)) / 2
                    if los and d <= rng_l - 2 and abs(wrap(math.atan2(tp[1] - y, tp[0] - x) - me["aim"][0])) < cone * 0.8 \
                            and me["energy"] >= 6:
                        fire1 = True
                elif kind1 == "laser":
                    charge = w1.get("charge", 0.0)
                    need = s1.get("charge", 0.75) - 0.04
                    rng_s = s1.get("range", 280.0)
                    if los and d < rng_s - 10 and d > 12:
                        if charge < need:
                            fire1 = True  # keep charging
                        else:
                            ydir, pdir, ddl = self.lead_aim(me, center, [0, 0, 0], 1e6)
                            if self.aim_err(me, ydir, pdir) * ddl < 2.6:
                                fire1 = False  # release now
                            else:
                                fire1 = True  # hold until aligned (auto-fires after hold_s)
                elif kind1 == "mortar":
                    spd = s1.get("speed", 72.0)
                    g = self.cfg["weapons"]["mortar"].get("gravity", 22.0)
                    tspd = hyp(tv[0], tv[1])
                    if 24 < d < 200 and (not los or tspd < 3.0 or d > 110):
                        ma = self.mortar_aim(me, tp, tv, spd, g)
                        if ma is not None:
                            aim = [ma[0], ma[1]]
                            used_mortar = True
                            me_err = self.aim_err(me, ma[0], ma[1])
                            fire0 = False
                            if me_err * ma[2] < 4.0:
                                fire1 = True
            elif w1 is not None and kind1 == "laser" and w1.get("charge", 0.0) > 0:
                fire1 = True
            if kind1 == "laser" and w1 is not None and w1.get("charge", 0.0) > 0 and not fire1:
                fire1 = False

            # movement
            if not los and not used_mortar:
                dirv = self.nav_to(me, tp[0], tp[1])
                mv = dirv
                label = "hunt"
            else:
                ux, uy = unit(tp[0] - x, tp[1] - y)
                px_, py_ = -uy, ux
                if t >= self.strafe_until:
                    if self.rng.random() < 0.7:
                        self.strafe = -self.strafe
                    self.strafe_until = t + self.rng.uniform(0.35, 1.0)
                if d > hi:
                    rad = 1.0
                elif d < lo:
                    rad = -1.0
                else:
                    rad = 0.0
                if hpf < 0.35 and d < hi + 30:
                    rad = -1.0
                vx_ = px_ * self.strafe * 0.9 + ux * rad * 0.8
                vy_ = py_ * self.strafe * 0.9 + uy * rad * 0.8
                mv = (vx_, vy_)
                label = "fight %d" % target["id"]
            if me["energy"] > 55 and hpf < 0.45 and d < 90:
                boost = True
            if kind1 == "lightning" and d > hi + 12 and me["energy"] > 65 and los:
                boost = True
            if aim is None:
                aim = [yaw, pitch]

        # ---------------- no combat target: loot / zone / idle
        else:
            goal = None
            shoot = None  # (pos, vel, kind)
            # supply drops
            for dr in obs.get("drops", []):
                dp = dr["pos"]
                dd = hyp(dp[0] - x, dp[1] - y)
                travel = dd / 11.0
                if dr.get("lands_in", 0) <= travel + 4 and dd < 380 and hyp(dp[0] - rc[0], dp[1] - rc[1]) < rr + 25:
                    if goal is None or dd < goal[3]:
                        goal = ("drop", dp[0], dp[1], dd, dr)
            if goal is None and not zone_urgent:
                want_slot = w1 is None
                cands = []
                for k, (kind, sub, pos, seen) in self.items.items():
                    d = hyp(pos[0] - x, pos[1] - y)
                    if hyp(pos[0] - rc[0], pos[1] - rc[1]) > rr - 12:
                        continue
                    if kind == "pickup":
                        if sub == "gun" or abs(pos[2] - z) > 7:
                            continue
                        r_new = WRANK.get(sub, 0)
                        r_old = WRANK.get(kind1, 0) if kind1 else 0
                        if kind1 == sub or (w1 is not None and r_new <= r_old):
                            continue
                        if d > 200:
                            continue
                        if self.nav is not None and self.nav.blocked(pos[0], pos[1]):
                            continue
                        cands.append((d - 70, "pickup", k, pos))
                    elif kind == "cube":
                        if d > 130 or abs(pos[2] - z) > 7:
                            continue
                        if self.nav is not None and self.nav.blocked(pos[0], pos[1]):
                            continue
                        cands.append((d - 45, "cube", k, pos))
                    elif kind == "crate":
                        if d > 120 or hpf < 0.7:
                            continue
                        cands.append((d - 22, "crate", k, pos))
                for m in npcs:
                    d = hyp(m["pos"][0] - x, m["pos"][1] - y)
                    if d < 130 and hpf > 0.7:
                        cands.append((d - 10, "npc", m["id"], m["pos"]))
                if cands:
                    cands.sort(key=lambda c: c[0])
                    c = cands[0]
                    goal = (c[1], c[3][0], c[3][1], hyp(c[3][0] - x, c[3][1] - y), c)

            if zone_urgent:
                v = self.nav_to(me, rc[0], rc[1])
                mv = v
                boost = (not me["in_zone"]) and me["energy"] > 20
                label = "zone"
            elif goal is not None:
                kind = goal[0]
                gx, gy, gd = goal[1], goal[2], goal[3]
                if kind == "drop":
                    ring = goal[4].get("radius", 8.0)
                    if gd < ring - 3:
                        mv = None
                    else:
                        mv = self.nav_to(me, gx, gy)
                    label = "drop"
                elif kind in ("crate", "npc"):
                    pos = goal[4][3]
                    if kind == "npc":
                        npc = next((m for m in npcs if m["id"] == goal[4][2]), None)
                        tvv = npc["vel"] if npc else [0, 0, 0]
                        cz = pos[2] + 1.75
                    else:
                        tvv = [0, 0, 0]
                        cz = pos[2] + 1.2
                    shoot = ((pos[0], pos[1], cz), tvv)
                    if gd > 38 or not self.los(me, pos):
                        mv = self.nav_to(me, gx, gy)
                    else:
                        ux, uy = unit(pos[0] - x, pos[1] - y)
                        if t >= self.strafe_until:
                            self.strafe = -self.strafe
                            self.strafe_until = t + self.rng.uniform(0.6, 1.4)
                        mv = (-uy * self.strafe * 0.6, ux * self.strafe * 0.6)
                    label = kind
                else:
                    mv = self.nav_to(me, gx, gy)
                    label = kind
            elif soft_out:
                tx = rc[0] + (x - rc[0]) * 0.45 * rr / max(dzone, 1.0)
                ty = rc[1] + (y - rc[1]) * 0.45 * rr / max(dzone, 1.0)
                mv = self.nav_to(me, tx, ty)
                label = "reposition"
            else:
                mv = None
                label = "hold"

            if shoot is not None:
                spd = w0["stats"]["speed"] if w0 and "stats" in w0 else 252.0
                yaw, pitch, dd = self.lead_aim(me, shoot[0], shoot[1], spd)
                aim = [yaw, pitch]
                gun_ok = w0 is not None and w0.get("ammo", 1) > 0 and w0.get("reload", 0) <= 0
                if gun_ok and dd < 150 and self.aim_err(me, yaw, pitch) * dd < 2.0:
                    fire0 = True

        # ---------------- dodge override
        if dodge is not None:
            mv = (dodge[0], dodge[1])
            if dodge[2] and me["energy"] > 20:
                boost = True
            label = "dodge"

        # ---------------- weapon pickup (when no immediate danger)
        near_enemy = target is not None and hyp(target["pos"][0] - x, target["pos"][1] - y) < 45
        if not near_enemy:
            for p in obs["pickups"]:
                if hyp(p["pos"][0] - x, p["pos"][1] - y) < 3.2 and abs(p["pos"][2] - z) < 3.5 and p["kind"] != "gun":
                    r_new = WRANK.get(p["kind"], 0)
                    if w1 is None and (kind1 != p["kind"]):
                        a["take"] = p["id"]
                        break
                    if w1 is not None and kind1 != p["kind"] and r_new > WRANK.get(kind1, 0) \
                            and not (kind1 == "laser" and w1.get("charge", 0) > 0):
                        a["take"] = p["id"]
                        a["drop"] = 1
                        break

        # ---------------- aim when nothing to shoot
        if aim is None:
            sound = None
            for s in obs.get("sounds", []):
                if s["kind"] in ("gun", "laser", "mortar", "lightning", "laser_charge", "step", "boost", "jet") and \
                        s["dist"] in ("near", "mid") and s["age"] < 0.5:
                    sound = s
                    break
            if self.dmg_bearing is not None and t - self.dmg_t < 1.5:
                aim = [self.dmg_bearing, 0.0]
            elif sound is not None:
                aim = [sound["bearing"], 0.0]
            elif mv is not None and hyp(mv[0], mv[1]) > 0.1:
                aim = [math.atan2(mv[1], mv[0]), 0.0]
            else:
                self.scan = wrap(self.scan + 0.5)
                aim = [self.scan, 0.0]

        # ---------------- stuck detection
        self.hist.append((t, x, y))
        while self.hist and t - self.hist[0][0] > 1.2:
            self.hist.pop(0)
        moving_cmd = mv is not None and hyp(mv[0], mv[1]) > 0.5
        if moving_cmd and len(self.hist) >= 8 and t >= self.unstick_until:
            h0 = self.hist[0]
            if hyp(x - h0[1], y - h0[2]) < 2.0 and t - h0[0] > 0.9:
                self.unstick_until = t + 1.0
                ang = self.rng.uniform(0, TAU)
                self.unstick_dir = (math.cos(ang), math.sin(ang))
                self.path = None
                self.path_goal = -2
                self.search = None
        if t < self.unstick_until:
            mv = self.unstick_dir
            jump = True
            boost = me["energy"] > 15
            label = "unstick"
        elif mv is not None and label not in ("dodge",):
            mv = self.safe_dir(me, mv[0], mv[1])

        # ---------------- reload
        if w0 is not None and w0.get("reload", 0) <= 0:
            if w0.get("ammo", 12) <= 0 or (w0.get("ammo", 12) <= 5 and (target is None or (tdist is not None and tdist > 90))):
                reload_now = True

        # ---------------- assemble
        if mv is None:
            a["move"] = [0.0, 0.0]
        else:
            m = hyp(mv[0], mv[1])
            if m > 1.0:
                mv = (mv[0] / m, mv[1] / m)
            a["move"] = [round(mv[0], 4), round(mv[1], 4)]
        a["boost"] = bool(boost)
        a["jump"] = bool(jump)
        a["aim"] = [round(aim[0], 4), round(aim[1], 4)]
        a["fire"] = [bool(fire0), bool(fire1)]
        if reload_now:
            a["reload"] = True
        mf.draw.label("%s hp%d" % (label, me["hp"]))
        return a


if __name__ == "__main__":
    mf.run(Bot)