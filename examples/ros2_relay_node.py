#!/usr/bin/env python3
"""EXAMPLE ONLY: wrap VelocityRelay + HomingGate in a ROS 2 node.

Subscribes to a JointTrajectory setpoint stream and a JointState feedback
topic, and calls `send_speeds(list_of_rad_s)` -- replace that with your arm's
velocity API (Kinova: Base.SendJointSpeedsCommand on ONE session, closed on
every exit path). rclpy is imported lazily so the kit itself never needs ROS.

    python3 examples/ros2_relay_node.py --ros-args -p arm:=left
"""
import math
import time


def main():
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from trajectory_msgs.msg import JointTrajectory

    from sim2real_gap import HomingGate, VelocityRelay

    class RelayNode(Node):
        def __init__(self):
            super().__init__("sim2real_relay")
            self.declare_parameter("arm", "left")
            self.declare_parameter("rate_hz", 30.0)
            self.declare_parameter("kp", 0.5)
            self.declare_parameter("vmax_rad_s", 0.05)
            self.declare_parameter("deadband_deg", 0.25)
            self.declare_parameter("watchdog_s", 0.5)
            self.declare_parameter("gate_rad", 0.05)
            arm = self.get_parameter("arm").value
            self.relay = VelocityRelay(
                kp=self.get_parameter("kp").value,
                vmax_rad_s=self.get_parameter("vmax_rad_s").value,
                deadband_rad=math.radians(self.get_parameter("deadband_deg").value),
                watchdog_s=self.get_parameter("watchdog_s").value,
                continuous_idx=(0, 2, 4, 6))
            self.gate = HomingGate(self.get_parameter("gate_rad").value, (0, 2, 4, 6))
            self.enabled = False
            self.actual = None
            self.create_subscription(JointState, "/real/joint_states", self._js, 10)
            self.create_subscription(JointTrajectory, "/real/%s_arm_controller/joint_trajectory" % arm, self._traj, 10)
            self.timer = self.create_timer(1.0 / self.get_parameter("rate_hz").value, self._tick)
            self._last_log = time.monotonic()

        def _js(self, m):
            self.actual = list(m.position)

        def _traj(self, m):
            if not m.points:
                return
            p = m.points[-1]
            if self.actual is not None and not self.enabled:
                ok, j, err = self.gate.check(p.positions, self.actual)
                if not ok:
                    self.get_logger().warn(self.gate.explain(p.positions, self.actual), throttle_duration_sec=2.0)
                    return
                self.enabled = True
                self.get_logger().info("gate passed; relay enabled")
            self.relay.set_target(p.positions, list(p.velocities) or None)

        def _tick(self):
            if self.actual is None or not self.enabled:
                return
            speeds = self.relay.step(self.actual)
            t0 = time.monotonic()
            self.send_speeds(speeds)                       # <- your arm API here
            self.relay.record_send_latency((time.monotonic() - t0) * 1000.0)
            if time.monotonic() - self._last_log > 5.0:
                self._last_log = time.monotonic()
                self.get_logger().info("achieved %.1f Hz, latency %s, %s" % (
                    self.relay.achieved_rate_hz or 0.0, self.relay.latency_summary_ms(), self.relay.last_reason))

        def send_speeds(self, speeds):
            pass                                            # replace

    rclpy.init()
    n = RelayNode()
    try:
        rclpy.spin(n)
    finally:
        n.send_speeds([0.0] * 7)                             # zero on every exit path
        n.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
