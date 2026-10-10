"""Teacher-driven bot: the scripted teacher from training/teacher.py drives the car.

NOT the ML student. Used to watch the teacher and (later) to record demonstrations.
v2: adds lightweight, bounded timing stats (printed, never written per frame).
"""
from collections import deque
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))       # observation_contract.py, prediction.py
sys.path.insert(0, str(REPO / "training"))  # teacher.py

from rlbot import flat
from rlbot.managers import Bot

from observation_contract import basis
from prediction import find_slice_at_time
from teacher import Teacher


def relative(delta, axes):
    """Project a world-space offset onto the car's forward/right/up axes."""
    return [sum(d * a for d, a in zip(delta, axis)) for axis in axes]


def build_state(packet, index, prediction):
    """Turn a live packet into the plain dict the teacher expects."""
    car = packet.players[index]
    p = car.physics
    pos = [p.location.x, p.location.y, p.location.z]
    vel = [p.velocity.x, p.velocity.y, p.velocity.z]
    axes = basis([p.rotation.pitch, p.rotation.yaw, p.rotation.roll])
    state = dict(
        ball_present=bool(packet.balls),
        ball_rel=None,
        pred_rel=None,
        speed=sum(v * v for v in vel) ** 0.5,
        on_ground=car.air_state == flat.AirState.OnGround,
    )
    if packet.balls:
        b = packet.balls[0].physics.location
        state["ball_rel"] = relative([b.x - pos[0], b.y - pos[1], b.z - pos[2]], axes)
        if prediction is not None and prediction.slices:
            sl = find_slice_at_time(prediction, packet.match_info.seconds_elapsed + 2)
            if sl is not None:
                q = sl.physics.location
                state["pred_rel"] = relative([q.x - pos[0], q.y - pos[1], q.z - pos[2]], axes)
    return state


def to_controls(mode, steer):
    """Same 4-mode decoding as the competition (src/runtime.py decode). No boost/handbrake."""
    c = flat.ControllerState()
    if mode == 1:
        c.throttle = 1.0
        c.steer = steer
    elif mode == 2:
        c.jump = True
    elif mode == 3:
        c.pitch = -1.0
        c.jump = True
    return c


class Timing:
    """Bounded timing stats (last 600 callbacks). Prints one line per REPORT_EVERY callbacks."""
    REPORT_EVERY = 1200

    def __init__(self):
        self.compute = deque(maxlen=600)    # seconds spent in OUR code per callback
        self.game_gap = deque(maxlen=600)   # game-clock seconds between callbacks
        self.wall_gap = deque(maxlen=600)   # real seconds between callbacks
        self.n = 0
        self.last_game = None
        self.last_wall = None

    def record(self, game_t, compute_s):
        wall = time.perf_counter()
        self.compute.append(compute_s)
        if self.last_game is not None and game_t > self.last_game:
            self.game_gap.append(game_t - self.last_game)
            self.wall_gap.append(wall - self.last_wall)
        self.last_game, self.last_wall = game_t, wall
        self.n += 1
        if self.n % self.REPORT_EVERY == 0:
            self.report()

    @staticmethod
    def _stats(values):
        if not values:
            return "n/a"
        v = sorted(values)
        return "mean %.1f / p95 %.1f / max %.1f ms" % (
            1000 * sum(v) / len(v), 1000 * v[int(0.95 * (len(v) - 1))], 1000 * v[-1])

    def report(self):
        print("TIMING callbacks=%d | our code: %s | game gap: %s | real gap: %s" % (
            self.n, self._stats(self.compute), self._stats(self.game_gap),
            self._stats(self.wall_gap)), flush=True)


class TeacherBot(Bot):
    def initialize(self):
        self.teacher = Teacher()
        self.timing = Timing()

    def get_output(self, packet):
        start = time.perf_counter()
        state = build_state(packet, self.index, self.ball_prediction)
        game_t = float(packet.match_info.seconds_elapsed)
        mode, steer = self.teacher.decide(state, game_t)
        controls = to_controls(mode, steer)
        self.timing.record(game_t, time.perf_counter() - start)
        return controls

    def retire(self):
        self.timing.report()
        super().retire()


if __name__ == "__main__":
    TeacherBot("aceteam/teacher").run()