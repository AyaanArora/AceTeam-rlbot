"""Teacher-driven bot that also RECORDS demonstrations (v3). NOT the ML student.

Each callback logs the exact 13 student inputs (via src/observation_contract.live_adapter,
same as src/runtime.py) and the teacher label. Rows are buffered and written in bursts to
datasets/teacher_v2/*.csv (git-ignored). One file = one bot run = one match stream.
"""
from collections import deque
from datetime import datetime
from pathlib import Path
import os
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))       # observation_contract.py, prediction.py
sys.path.insert(0, str(REPO / "training"))  # teacher.py

from rlbot import flat
from rlbot.managers import Bot

from observation_contract import basis, live_adapter
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
        line = "TIMING callbacks=%d | our code: %s | game gap: %s | real gap: %s" % (
            self.n, self._stats(self.compute), self._stats(self.game_gap),
            self._stats(self.wall_gap))
        print(line, flush=True)
        try:                                   # also keep it in logs/ so no console is needed
            folder = REPO / "logs"
            folder.mkdir(exist_ok=True)
            with (folder / "teacher_timing.txt").open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass


HEADER = ("T," + ",".join("o%d" % i for i in range(13)) + ",mode,steer,on_ground\n")


class Recorder:
    """Buffers rows in memory; appends to a CSV every FLUSH_EVERY rows (cheap bursts)."""
    FLUSH_EVERY = 2000

    def __init__(self):
        folder = REPO / "datasets" / "teacher_v2"
        folder.mkdir(parents=True, exist_ok=True)
        self.path = folder / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_%d.csv" % os.getpid())
        self.rows, self.written, self.skipped = [], 0, 0
        self.prev_T, self.prev_mode, self.prev_steer = None, 0, 0.0

    def log(self, packet, index, prediction, T, mode, steer, on_ground):
        selected, first = None, None
        if prediction is not None and prediction.slices:
            first = prediction.slices[0].game_seconds
            selected = find_slice_at_time(prediction, T + 2)
        ctx = dict(previous_elapsed=self.prev_T, previous_mode=self.prev_mode,
                   previous_steer=self.prev_steer, prediction_valid=selected is not None,
                   first_time=first)
        try:
            obs = list(live_adapter(packet, index, ctx, selected))[:13]
        except Exception as error:  # e.g. duplicate clock: skip row, never break driving
            self.skipped += 1
            if self.skipped == 1:
                print("RECORDER first skipped row:", type(error).__name__, error, flush=True)
            return
        self.prev_T, self.prev_mode, self.prev_steer = T, mode, steer
        self.rows.append("%.6f," % T + ",".join("%.9g" % x for x in obs)
                         + ",%d,%.6f,%d" % (mode, steer, int(on_ground)))
        if len(self.rows) >= self.FLUSH_EVERY:
            self.flush()

    def flush(self):
        if not self.rows:
            return
        try:                                   # a file lock (e.g. OneDrive) must not stop the bot
            new = not self.path.exists()
            with self.path.open("a", encoding="utf-8") as f:
                if new:
                    f.write(HEADER)
                f.write("\n".join(self.rows) + "\n")
        except OSError as error:
            print("RECORDER write failed, will retry:", error, flush=True)
            return
        self.written += len(self.rows)
        self.rows = []
        print("RECORDER rows=%d skipped=%d file=%s" % (self.written, self.skipped, self.path.name), flush=True)


class TeacherBot(Bot):
    def initialize(self):
        self.teacher = Teacher()
        self.timing = Timing()
        self.recorder = Recorder()

    def get_output(self, packet):
        start = time.perf_counter()
        state = build_state(packet, self.index, self.ball_prediction)
        game_t = float(packet.match_info.seconds_elapsed)
        mode, steer = self.teacher.decide(state, game_t)
        controls = to_controls(mode, steer)
        self.recorder.log(packet, self.index, self.ball_prediction, game_t, mode, steer,
                          state["on_ground"])
        self.timing.record(game_t, time.perf_counter() - start)
        return controls

    def retire(self):
        self.recorder.flush()
        self.timing.report()
        super().retire()


if __name__ == "__main__":
    TeacherBot("aceteam/teacher").run()