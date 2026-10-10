"""Scripted teacher v2: creates training labels only. NOT part of the submitted bot.

Modes: 0 Neutral, 1 Chase, 2 Jump, 3 Front dodge. Steering is only used in Chase.
Inputs it uses: ball_rel / pred_rel (car-relative UU, same data the student sees),
speed, plus on_ground (NOT a student input; the student must infer it).
All thresholds live in Config so they can be tuned from live observation.
"""
import math
from dataclasses import dataclass

NEUTRAL, CHASE, JUMP, DODGE = 0, 1, 2, 3


@dataclass
class Config:
    steer_gain: float = 5.0        # same as the RLBot example bot
    far_dist: float = 1500.0       # beyond this aim at the 2 s prediction (example bot rule)
    reach_up: float = 300.0        # ball higher than this (rel. to car) is out of ground-jump reach:
                                   # aim at the prediction instead of driving underneath it
    hit_up_min: float = 150.0      # ball between hit_up_min and reach_up -> jump to it
    hit_lead: float = 0.45         # start the jump when time-to-contact (fwd/speed) <= this (s)
    hit_min_lead: float = 0.15     # ...but not when already almost there (s)
    hit_side: float = 120.0        # max sideways offset (UU) for a jump hit
    hit_min_speed: float = 300.0
    jump_hold_hit: float = 0.20    # hold jump longer for height (s)
    flip_min_speed: float = 600.0  # speed-flip window: below, accelerating is cheaper
    flip_max_speed: float = 1300.0 # above, near top throttle speed (~1410)
    flip_min_dist: float = 2000.0  # only flip for speed when the ball is far
    align_angle: float = 0.15      # rad; must face the target to speed-flip
    flip_jump: float = 0.05        # reference example timings: jump, release, dodge
    flip_gap: float = 0.05
    flip_dodge: float = 0.20
    cooldown: float = 2.0          # min seconds between sequence starts (stops flip spam)


def _clamp(x, lo=-1.0, hi=1.0):
    return max(lo, min(hi, x))


class Teacher:
    """One Teacher per bot. Remembers the running jump/flip sequence."""

    def __init__(self, config=None):
        self.cfg = config or Config()
        self.seq = None            # (start_time, [(duration, mode), ...])
        self.last_start = -1e9
        self.last_now = None

    def decide(self, s, now):
        """s: ball_present, ball_rel, pred_rel, speed, on_ground. Returns (mode, steer)."""
        c = self.cfg
        if self.last_now is not None and now < self.last_now:   # clock restarted (new match)
            self.seq, self.last_start = None, -1e9
        self.last_now = now
        if not s["ball_present"]:
            self.seq = None
            return NEUTRAL, 0.0

        fwd, right, up = s["ball_rel"]
        dist = math.sqrt(fwd * fwd + right * right + up * up)
        pred = s.get("pred_rel")
        use_pred = pred is not None and (dist > c.far_dist or up > c.reach_up)
        tx, ty = (pred[0], pred[1]) if use_pred else (fwd, right)
        angle = math.atan2(ty, tx)
        steer = _clamp(angle * c.steer_gain)

        if self.seq is not None:
            start, phases = self.seq
            t = now - start
            for duration, mode in phases:
                if t < duration:
                    return mode, (steer if mode == CHASE else 0.0)
                t -= duration
            self.seq = None                                   # finished: back to normal driving

        if s["on_ground"] and now - self.last_start >= c.cooldown:
            phases = self._pick(s, fwd, right, up, dist, angle)
            if phases:
                self.seq, self.last_start = (now, phases), now
                mode = phases[0][1]
                return mode, 0.0
        return CHASE, steer

    def _pick(self, s, fwd, right, up, dist, angle):
        c, speed = self.cfg, s["speed"]
        if c.hit_up_min <= up <= c.reach_up and fwd > 0 and abs(right) <= c.hit_side \
                and speed >= c.hit_min_speed:
            tc = fwd / speed
            if c.hit_min_lead <= tc <= c.hit_lead:
                return [(c.jump_hold_hit, JUMP)]
        if up < c.hit_up_min and dist > c.flip_min_dist and abs(angle) < c.align_angle \
                and c.flip_min_speed <= speed <= c.flip_max_speed:
            return [(c.flip_jump, JUMP), (c.flip_gap, CHASE), (c.flip_dodge, DODGE)]
        return None