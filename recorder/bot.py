"""Teacher-driven bot: the scripted teacher from training/teacher.py drives the car.

This is NOT the ML student. It exists so we can watch the teacher play before
using it to create training data. Step 1: driving only, no recording yet.
"""
from pathlib import Path
import sys

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
    """Same 4-mode decoding as the competition (src/runtime.py decode)."""
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


class TeacherBot(Bot):
    def initialize(self):
        self.teacher = Teacher()

    def get_output(self, packet):
        state = build_state(packet, self.index, self.ball_prediction)
        mode, steer = self.teacher.decide(state, float(packet.match_info.seconds_elapsed))
        return to_controls(mode, steer)


if __name__ == "__main__":
    TeacherBot("aceteam/teacher").run()