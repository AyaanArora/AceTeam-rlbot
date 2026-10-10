"""Offline checks for teacher.py (state machine only, NOT game physics). Run from repo root."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from teacher import Teacher, NEUTRAL, CHASE, JUMP, DODGE

failures = []


def check(name, got, ok):
    print("PASS " if ok else "FAIL ", name, "->", got)
    if not ok:
        failures.append(name)


def st(ball=(3000.0, 0.0, 76.0), pred=None, speed=0.0, ground=True, present=True):
    return dict(ball_present=present, ball_rel=ball, pred_rel=pred, speed=speed, on_ground=ground)


def one(state, now=0.0):
    return Teacher().decide(state, now)


r = one(st(present=False)); check("no ball", r, r == (NEUTRAL, 0.0))
r = one(st()); check("ball ahead", r, r[0] == CHASE and abs(r[1]) < 1e-6)
r = one(st(ball=(2000.0, 100.0, 76.0), speed=1500.0)); check("slightly right", r, r[0] == CHASE and 0.2 < r[1] < 0.3)
r = one(st(ball=(2000.0, -1000.0, 76.0), speed=1500.0)); check("far left", r, r == (CHASE, -1.0))
r = one(st(ball=(-300.0, 0.0, 76.0))); check("ball behind", r, r[0] == CHASE and abs(r[1]) == 1.0)
r = one(st(ball=(800.0, 0.0, 76.0), pred=(800.0, 1500.0, 76.0))); check("near ignores prediction", r, r[0] == CHASE and abs(r[1]) < 1e-6)
r = one(st(pred=(3000.0, 1500.0, 76.0))); check("far uses prediction", r, r[0] == CHASE and r[1] > 0.5)

# elevated, out of reach -> do not drive under it; aim at prediction (to the right here)
r = one(st(ball=(400.0, 0.0, 500.0), pred=(1500.0, 600.0, 76.0), speed=900.0))
check("elevated unreachable uses prediction, no jump", r, r == (CHASE, 1.0))
# ball on a wall/very high: same rule
r = one(st(ball=(500.0, 0.0, 1200.0), pred=(1200.0, -300.0, 76.0), speed=900.0))
check("ball very high uses prediction", r, r[0] == CHASE and r[1] < 0)

# elevated, reachable, in time window -> one jump, then back to Chase
t = Teacher(); hit = st(ball=(300.0, 0.0, 200.0), speed=700.0)   # tc = 0.43 s
m = [t.decide(hit, x)[0] for x in (0.0, 0.1, 0.19, 0.25)]
check("elevated jump then chase", m, m == [JUMP, JUMP, JUMP, CHASE])
r = one(st(ball=(1000.0, 0.0, 200.0), speed=700.0)); check("elevated but too far: keep driving", r, r[0] == CHASE)
r = one(st(ball=(300.0, 400.0, 200.0), speed=700.0)); check("elevated but off to side: no jump", r, r[0] == CHASE)

# ground ball close: just drive, no flip spam
t = Teacher(); modes = [t.decide(st(ball=(300.0, 0.0, 76.0), speed=900.0), x * 0.05)[0] for x in range(60)]
check("close ground ball: no flips", set(modes), set(modes) == {CHASE})

# speed flip sequence, then no retrigger during cooldown, allowed again after
t = Teacher(); far = st(speed=800.0)
m = [t.decide(far, x)[0] for x in (0.0, 0.04, 0.07, 0.12, 0.29, 0.31)]
check("flip sequence", m, m == [JUMP, JUMP, CHASE, DODGE, DODGE, CHASE])
later = [t.decide(far, x)[0] for x in (0.5, 1.0, 1.5, 1.9)]
check("no retrigger in cooldown", later, set(later) == {CHASE})
r = t.decide(far, 2.1); check("allowed after cooldown", r, r[0] == JUMP)

r = one(st(speed=800.0, ground=False)); check("no flip in air", r, r[0] == CHASE)
r = one(st(speed=100.0)); check("too slow to flip", r, r[0] == CHASE)
r = one(st(speed=1500.0)); check("too fast to flip", r, r[0] == CHASE)
r = one(st(ball=(3000.0, 1500.0, 76.0), speed=800.0)); check("not aligned: no flip", r, r[0] == CHASE)
t = Teacher(); t.decide(far, 100.0)
r = t.decide(far, 5.0); check("clock restart resets cooldown", r, r[0] == JUMP)

# crude kickoff run: speed ramps, ball distance shrinks. Counts sequence starts only.
t, dist, speed, starts, prev, now = Teacher(), 3400.0, 0.0, [], CHASE, 0.0
while dist > 400 and now < 8:
    airborne = t.seq is not None or (starts and now - starts[-1] < 1.2)
    mode, _ = t.decide(st(ball=(dist, 0.0, 76.0), speed=speed, ground=not airborne), now)
    if mode == JUMP and prev != JUMP:
        starts.append(now); speed += 300
    prev = mode
    speed = min(1410.0, speed + (0 if airborne else 700 * (1 - speed / 1500) / 60 * 2.2))
    dist -= speed / 60; now += 1 / 60
check("kickoff run flips (state machine only)", [round(x, 2) for x in starts],
      len(starts) <= 2 and all(b - a >= 2.0 for a, b in zip(starts, starts[1:])))

if failures:
    print("\nFAILED:", failures); sys.exit(1)
print("\nALL TESTS PASSED")