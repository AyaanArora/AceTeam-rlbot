"""Offline checks for teacher.py. Run from the repo root."""
import sys
from teacher import Teacher, NEUTRAL, CHASE, JUMP, DODGE

failures = []


def check(name, got, ok):
    print(("PASS " if ok else "FAIL "), name, "->", got)
    if not ok:
        failures.append(name)


def state(ball=(3000.0, 0.0, 93.0), pred=None, speed=0.0, ground=True, present=True):
    return dict(ball_present=present, ball_rel=ball, pred_rel=pred,
                speed=speed, on_ground=ground)


# 1. No ball -> Neutral
r = Teacher().decide(state(present=False), 0.0)
check("no ball", r, r == (NEUTRAL, 0.0))

# 2. Ball straight ahead, standing still -> Chase, steer 0
r = Teacher().decide(state(), 0.0)
check("ball ahead, slow", r, r[0] == CHASE and abs(r[1]) < 1e-6)

# 3. Ball far to the right -> Chase, full right steer
r = Teacher().decide(state(ball=(2000.0, 1000.0, 93.0), speed=1500.0), 0.0)
check("ball right", r, r[0] == CHASE and r[1] == 1.0)

# 4. Ball far to the left -> Chase, full left steer
r = Teacher().decide(state(ball=(2000.0, -1000.0, 93.0), speed=1500.0), 0.0)
check("ball left", r, r[0] == CHASE and r[1] == -1.0)

# 5. Ball slightly right -> small positive steer
r = Teacher().decide(state(ball=(2000.0, 100.0, 93.0), speed=1500.0), 0.0)
check("ball slightly right", r, r[0] == CHASE and 0.2 < r[1] < 0.3)

# 6. Ball far: aim at prediction (to the right), not at ball (ahead)
r = Teacher().decide(state(pred=(3000.0, 1500.0, 93.0)), 0.0)
check("far uses prediction", r, r[0] == CHASE and r[1] > 0.5)

# 7. Ball near: ignore prediction
r = Teacher().decide(state(ball=(800.0, 0.0, 93.0), pred=(800.0, 1500.0, 93.0)), 0.0)
check("near ignores prediction", r, r[0] == CHASE and abs(r[1]) < 1e-6)

# 8. Ball behind and close -> still Chase (turns)
r = Teacher().decide(state(ball=(-300.0, 0.0, 76.0)), 0.0)
check("ball behind", r, r[0] == CHASE)

# 9. Full flip sequence (far ball, aligned, speed 800)
t = Teacher()
far = state(speed=800.0)
modes = [t.decide(far, 0.00)[0], t.decide(far, 0.05)[0], t.decide(far, 0.10)[0],
         t.decide(far, 0.15)[0], t.decide(far, 0.50)[0]]
check("flip sequence", modes, modes == [JUMP, JUMP, NEUTRAL, DODGE, NEUTRAL])
r = t.decide(state(speed=1500.0), 1.3)
check("chase after flip", r, r[0] == CHASE)

# 10. Close ball ahead -> flip starts
r = Teacher().decide(state(ball=(300.0, 0.0, 76.0), speed=900.0), 0.0)
check("close flip", r, r[0] == JUMP)

# 11. Same far situation but car in the air -> no flip
r = Teacher().decide(state(speed=800.0, ground=False), 0.0)
check("no flip in air", r, r[0] == CHASE)

# 12. Too slow for a far flip -> keep accelerating
r = Teacher().decide(state(speed=100.0), 0.0)
check("too slow to flip", r, r[0] == CHASE)

if failures:
    print("\nFAILED:", failures)
    sys.exit(1)
print("\nALL TESTS PASSED")