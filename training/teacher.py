"""Scripted teacher: decides which of the 4 modes to use and how to steer.

Only used to create training labels. It is NOT part of the submitted bot.
The teacher may use information the student does not see (e.g. on_ground).

Modes: 0 Neutral, 1 Chase, 2 Jump, 3 Front dodge.
Steering: -1 (left) to +1 (right).
"""
import math

NEUTRAL, CHASE, JUMP, DODGE = 0, 1, 2, 3

# ---- Tunable constants (starting guesses, to be adjusted from observation) ----
STEER_GAIN = 5.0          # same style as the RLBot example bot
FAR_DIST = 1500.0         # beyond this, aim at the 2-second ball prediction
FAR_FLIP_DIST = 1800.0    # only do a speed flip if the ball is at least this far
FLIP_MIN_SPEED = 500.0    # below this, keep accelerating first
FLIP_MAX_SPEED = 1300.0   # above this, a flip is not worth the time
ALIGN_ANGLE = 0.25        # radians; must be facing the target to flip
CLOSE_FWD_MIN = 150.0     # close flip window: ball this far ahead...
CLOSE_FWD_MAX = 450.0
CLOSE_SIDE_MAX = 130.0    # ...and roughly in line
CLOSE_UP_MAX = 200.0      # ...and low enough to hit
CLOSE_MIN_SPEED = 300.0
# Flip timeline in seconds since the flip started
JUMP_END = 0.08
GAP_END = 0.13
DODGE_END = 0.23
RECOVER_END = 1.2


def _clamp(x, lo=-1.0, hi=1.0):
    return max(lo, min(hi, x))


class Teacher:
    """Create one Teacher per match. It remembers when a flip started."""

    def __init__(self):
        self.flip_start = None

    def decide(self, s, now):
        """s: dict with ball_present, ball_rel (fwd,right,up in UU),
        pred_rel (same or None), speed (UU/s), on_ground (bool).
        now: match time in seconds. Returns (mode, steer)."""
        if self.flip_start is not None:
            return self._flip_step(s, now)
        if not s["ball_present"]:
            return NEUTRAL, 0.0

        ball = s["ball_rel"]
        dist = math.sqrt(sum(v * v for v in ball))
        target = ball
        pred = s.get("pred_rel")
        if dist > FAR_DIST and pred is not None:
            target = pred

        angle = math.atan2(target[1], target[0])
        steer = _clamp(angle * STEER_GAIN)

        if s["on_ground"] and self._want_flip(s, ball, dist, angle):
            self.flip_start = now
            return JUMP, 0.0
        return CHASE, steer

    def _want_flip(self, s, ball, dist, angle):
        speed = s["speed"]
        fwd, right, up = ball
        far_flip = (dist > FAR_FLIP_DIST
                    and FLIP_MIN_SPEED <= speed <= FLIP_MAX_SPEED
                    and abs(angle) < ALIGN_ANGLE)
        close_flip = (CLOSE_FWD_MIN < fwd < CLOSE_FWD_MAX
                      and abs(right) < CLOSE_SIDE_MAX
                      and up < CLOSE_UP_MAX
                      and speed >= CLOSE_MIN_SPEED)
        return far_flip or close_flip

    def _flip_step(self, s, now):
        t = now - self.flip_start
        if t < JUMP_END:
            return JUMP, 0.0
        if t < GAP_END:
            return NEUTRAL, 0.0
        if t < DODGE_END:
            return DODGE, 0.0
        if t < RECOVER_END:
            return NEUTRAL, 0.0
        self.flip_start = None
        return self.decide(s, now)