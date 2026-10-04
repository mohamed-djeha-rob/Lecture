"""Animate a torque-controlled pendulum using a PD controller.

Run with: python inverted_pendulum_pd.py

The angle is measured from the downward vertical, so the downward position is
0 rad.
"""

from dataclasses import dataclass
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import RadioButtons, Slider


@dataclass(frozen=True)
class PendulumConfig:
    mass: float = 0.5
    length: float = 1.0
    damping: float = 0.05
    gravity: float = 9.81
    kp: float = 10.0
    kd: float = 2.5
    max_torque: float = 5.0
    setpoint: float = 0.0
    initial_angle: float = np.deg2rad(25.0)
    initial_angular_velocity: float = 0.0
    duration: float = 25.0
    dt: float = 0.01

    def __post_init__(self) -> None:
        if self.mass <= 0 or self.length <= 0 or self.gravity <= 0:
            raise ValueError("Mass, length, and gravity must be positive.")
        if self.duration <= 0 or self.dt <= 0 or self.max_torque <= 0:
            raise ValueError("Duration, time step, and max torque must be positive.")
        if self.damping < 0 or self.kp < 0 or self.kd < 0:
            raise ValueError("Damping and PD gains cannot be negative.")


def linearize_model(
    config: PendulumConfig,
    setpoint: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return state-space matrices (A, B) linearized around the setpoint.

    The state is (angle, angular velocity), with angle measured from downward.
    The input is the torque deviation from the equilibrium torque needed to
    hold the pendulum at the setpoint.
    """
    angle = config.setpoint if setpoint is None else setpoint
    inertia = config.mass * config.length**2
    state_matrix = np.array(
        (
            (0.0, 1.0),
            (-config.gravity / config.length * np.cos(angle), -config.damping / inertia),
        )
    )
    input_matrix = np.array(((0.0,), (1.0 / inertia,)))
    return state_matrix, input_matrix


def pole_placement_gains(
    config: PendulumConfig,
    desired_poles: Sequence[complex],
    setpoint: float | None = None,
) -> tuple[float, float]:
    """Return proportional and derivative gains for the requested closed-loop poles.

    The gains use the feedback law u = u_eq - kp * angle_error - kd * angular
    velocity, where u_eq balances gravity at the setpoint.
    """
    if len(desired_poles) != 2:
        raise ValueError("Exactly two desired poles are required.")
    first_pole, second_pole = (complex(pole) for pole in desired_poles)
    if not all(
        np.isfinite((pole.real, pole.imag)).all()
        for pole in (first_pole, second_pole)
    ):
        raise ValueError("Desired poles must be finite.")
    if first_pole.real >= 0 or second_pole.real >= 0:
        raise ValueError("Desired poles must have negative real parts.")
    real_poles = np.isclose(first_pole.imag, 0.0) and np.isclose(
        second_pole.imag, 0.0
    )
    conjugate_pair = np.isclose(first_pole, np.conjugate(second_pole))
    if not real_poles and not conjugate_pair:
        raise ValueError("Desired poles must be real or a complex-conjugate pair.")

    angle = config.setpoint if setpoint is None else setpoint
    inertia = config.mass * config.length**2
    pole_sum = (first_pole + second_pole).real
    pole_product = (first_pole * second_pole).real
    kp = inertia * (
        pole_product - config.gravity / config.length * np.cos(angle)
    )
    kd = inertia * -pole_sum - config.damping
    return float(kp), float(kd)


def _control_torque(
    config: PendulumConfig,
    state: np.ndarray,
    setpoint: float,
    gains: tuple[float, float] | None = None,
) -> float:
    angle, angular_velocity = state
    kp, kd = (config.kp, config.kd) if gains is None else gains
    inertia = config.mass * config.length**2
    equilibrium_torque = (
        inertia * config.gravity / config.length * np.sin(setpoint)
    )
    torque = (
        equilibrium_torque
        + kp * (setpoint - angle)
        - kd * angular_velocity
    )
    return float(np.clip(torque, -config.max_torque, config.max_torque))


def _state_derivative(
    config: PendulumConfig,
    state: np.ndarray,
    setpoint: float,
    gains: tuple[float, float] | None = None,
) -> np.ndarray:
    angle, angular_velocity = state
    inertia = config.mass * config.length**2
    angular_acceleration = (
        -config.gravity / config.length * np.sin(angle)
        + (
            _control_torque(config, state, setpoint, gains)
            - config.damping * angular_velocity
        )
        / inertia
    )
    return np.array((angular_velocity, angular_acceleration))


def simulate(
    config: PendulumConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return simulation times, angles, and applied control torques."""
    step_count = int(np.ceil(config.duration / config.dt))
    times = np.linspace(0.0, config.duration, step_count + 1)
    dt = times[1] - times[0]
    states = np.zeros((step_count + 1, 2), dtype=float)
    states[0] = (config.initial_angle, config.initial_angular_velocity)

    for index in range(step_count):
        state = states[index]
        k1 = _state_derivative(config, state, config.setpoint)
        k2 = _state_derivative(config, state + 0.5 * dt * k1, config.setpoint)
        k3 = _state_derivative(config, state + 0.5 * dt * k2, config.setpoint)
        k4 = _state_derivative(config, state + dt * k3, config.setpoint)
        states[index + 1] = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    torques = np.array(
        [_control_torque(config, state, config.setpoint) for state in states]
    )
    return times, states[:, 0], torques


def animate(config: PendulumConfig) -> None:
    nominal_matrix, input_matrix = linearize_model(config, setpoint=0.0)
    desired_poles = np.linalg.eigvals(
        nominal_matrix - input_matrix @ np.array(((config.kp, config.kd),))
    )
    real_poles = np.sort(desired_poles.real)
    pole_decay_1 = float(-real_poles[-1])
    pole_decay_2 = float(-real_poles[0])
    pole_frequency = float(np.max(np.abs(desired_poles.imag)))
    pole_mode = "Conjugate pair"
    if pole_frequency <= 1e-8:
        pole_mode = (
            "Double real pole"
            if np.isclose(pole_decay_1, pole_decay_2)
            else "Two distinct real poles"
        )
    current_pole_description = ""
    current_gains = pole_placement_gains(config, desired_poles, config.setpoint)
    step_count = int(np.ceil(config.duration / config.dt))
    times = np.linspace(0.0, config.duration, step_count + 1)
    dt = times[1] - times[0]
    states = np.zeros((step_count + 1, 2), dtype=float)
    states[0] = (config.initial_angle, config.initial_angular_velocity)
    angles = states[:, 0]
    angle_degrees = np.rad2deg(angles)
    torques = np.full(step_count + 1, np.nan)
    torques[0] = _control_torque(
        config, states[0], config.setpoint, current_gains
    )
    current_setpoint = config.setpoint
    setpoint_degrees = np.rad2deg(config.setpoint)
    radius = config.length
    current_step = 0
    frame_stride = max(1, int(round(0.02 / config.dt)))
    frames = list(range(0, len(times), frame_stride))
    if frames[-1] != len(times) - 1:
        frames.append(len(times) - 1)

    figure = plt.figure(figsize=(12, 10))
    layout = figure.add_gridspec(
        7,
        2,
        width_ratios=(1, 1.5),
        height_ratios=(1, 1, 0.3, 0.16, 0.16, 0.16, 0.16),
    )
    pendulum_ax = figure.add_subplot(layout[:2, 0])
    angle_ax = figure.add_subplot(layout[0, 1])
    torque_ax = figure.add_subplot(layout[1, 1])
    pole_mode_ax = figure.add_subplot(layout[2, :])
    setpoint_slider_ax = figure.add_subplot(layout[3, :])
    decay_1_slider_ax = figure.add_subplot(layout[4, :])
    decay_2_slider_ax = figure.add_subplot(layout[5, :])
    frequency_slider_ax = figure.add_subplot(layout[6, :])

    pendulum_ax.set(
        xlim=(-1.25 * radius, 1.25 * radius),
        ylim=(-1.25 * radius, 1.25 * radius),
        aspect="equal",
        title="Pendulum",
        xlabel="x [m]",
        ylabel="y [m]",
    )
    pendulum_ax.axhline(0, color="0.6", linewidth=1)
    pendulum_ax.plot(0, 0, "ko", markersize=7, label="Pivot")
    target_rod, = pendulum_ax.plot(
        [0, radius * np.sin(config.setpoint)],
        [0, -radius * np.cos(config.setpoint)],
        "--",
        color="tab:green",
        linewidth=2,
        label="Desired position",
    )
    rod, = pendulum_ax.plot([], [], "o-", color="tab:blue", linewidth=4, markersize=9)
    pendulum_ax.legend(loc="lower left")

    angle_limit = max(
        180.0,
        1.2 * max(float(np.max(np.abs(angle_degrees))), abs(setpoint_degrees)),
    )
    angle_ax.set(
        xlim=(0, config.duration),
        ylim=(-angle_limit, angle_limit),
        title="Angle response",
        xlabel="Time [s]",
        ylabel="Angle [deg]",
    )
    setpoint_line = angle_ax.axhline(
        setpoint_degrees,
        linestyle="--",
        color="tab:green",
        label="Setpoint",
    )
    angle_line, = angle_ax.plot([], [], color="tab:blue", label="Pendulum angle")
    angle_marker, = angle_ax.plot([], [], "o", color="tab:blue")
    angle_ax.legend(loc="upper right")
    angle_ax.grid(True, alpha=0.3)

    torque_ax.set(
        xlim=(0, config.duration),
        ylim=(-1.1 * config.max_torque, 1.1 * config.max_torque),
        xlabel="Time [s]",
        ylabel="Torque [N m]",
        title="PD control effort",
    )
    torque_ax.axhline(0, color="0.5", linewidth=0.8)
    torque_line, = torque_ax.plot([], [], color="tab:orange")
    torque_ax.grid(True, alpha=0.3)

    setpoint_slider = Slider(
        setpoint_slider_ax,
        "Set point [deg]",
        -180.0,
        180.0,
        valinit=setpoint_degrees,
    )
    decay_1_slider = Slider(
        decay_1_slider_ax,
        "First pole decay [1/s]",
        0.001,
        max(10.0, 2.0 * pole_decay_1),
        valinit=max(0.001, pole_decay_1),
        valstep=0.01,
    )
    decay_2_slider = Slider(
        decay_2_slider_ax,
        "Second pole decay [1/s]",
        0.001,
        max(10.0, 2.0 * pole_decay_2),
        valinit=max(0.001, pole_decay_2),
        valstep=0.01,
    )
    frequency_slider = Slider(
        frequency_slider_ax,
        "Pole frequency [rad/s]",
        0.0,
        max(10.0, 2.0 * pole_frequency),
        valinit=pole_frequency,
        valstep=0.01,
    )
    pole_mode_selector = RadioButtons(
        pole_mode_ax,
        (
            "Conjugate pair",
            "Double real pole",
            "Two distinct real poles",
        ),
        active=(
            "Conjugate pair",
            "Double real pole",
            "Two distinct real poles",
        ).index(pole_mode),
    )
    pole_mode_positions = np.array((0.04, 0.36, 0.70))
    pole_mode_selector._buttons.set_offsets(
        np.column_stack((pole_mode_positions, np.full(3, 0.5)))
    )
    for position, label in zip(
        pole_mode_positions, pole_mode_selector.labels
    ):
        label.set_position((position + 0.025, 0.5))
    pole_mode_ax.set_axis_off()

    def update_controller() -> None:
        nonlocal current_gains, current_pole_description
        if pole_mode == "Double real pole":
            current_poles = (
                complex(-pole_decay_1, 0.0),
                complex(-pole_decay_1, 0.0),
            )
            current_pole_description = (
                f"-{pole_decay_1:.2f} (double real pole)"
            )
        elif pole_mode == "Two distinct real poles":
            current_poles = (
                complex(-pole_decay_1, 0.0),
                complex(-pole_decay_2, 0.0),
            )
            current_pole_description = (
                f"-{pole_decay_1:.2f}, -{pole_decay_2:.2f} "
                "(distinct real poles)"
            )
        else:
            current_poles = (
                complex(-pole_decay_1, pole_frequency),
                complex(-pole_decay_1, -pole_frequency),
            )
            current_pole_description = (
                f"-{pole_decay_1:.2f} +/- {pole_frequency:.2f}j"
            )
        current_gains = pole_placement_gains(
            config, current_poles, current_setpoint
        )
        torques[current_step] = _control_torque(
            config, states[current_step], current_setpoint, current_gains
        )
        update(current_step)
        figure.canvas.draw_idle()

    def update(frame: int) -> tuple:
        nonlocal current_step
        for index in range(current_step, frame):
            state = states[index]
            k1 = _state_derivative(config, state, current_setpoint, current_gains)
            k2 = _state_derivative(
                config,
                state + 0.5 * dt * k1,
                current_setpoint,
                current_gains,
            )
            k3 = _state_derivative(
                config,
                state + 0.5 * dt * k2,
                current_setpoint,
                current_gains,
            )
            k4 = _state_derivative(
                config, state + dt * k3, current_setpoint, current_gains
            )
            states[index + 1] = state + (dt / 6.0) * (
                k1 + 2 * k2 + 2 * k3 + k4
            )
            torques[index + 1] = _control_torque(
                config, states[index + 1], current_setpoint, current_gains
            )
        current_step = frame
        angles[:] = states[:, 0]
        angle_degrees = np.rad2deg(angles)
        angle = angles[frame]
        x = radius * np.sin(angle)
        y = -radius * np.cos(angle)
        rod.set_data([0, x], [0, y])
        angle_line.set_data(times[: frame + 1], angle_degrees[: frame + 1])
        angle_marker.set_data([times[frame]], [angle_degrees[frame]])
        torque_line.set_data(times[: frame + 1], torques[: frame + 1])
        figure.suptitle(
            f"PD control  |  time = {times[frame]:.2f} s  |  "
            f"Kp = {current_gains[0]:.2f}, "
            f"Kd = {current_gains[1]:.2f}  |  "
            f"poles = {current_pole_description}"
        )
        return rod, angle_line, angle_marker, torque_line

    def update_setpoint(value: float) -> None:
        nonlocal current_setpoint
        current_setpoint = float(np.deg2rad(value))
        setpoint_limit = max(angle_limit, 1.2 * abs(value))
        angle_ax.set_ylim(-setpoint_limit, setpoint_limit)
        setpoint_line.set_ydata((value, value))
        target_rod.set_data(
            [0, radius * np.sin(current_setpoint)],
            [0, -radius * np.cos(current_setpoint)],
        )
        update_controller()

    def update_decay_1(value: float) -> None:
        nonlocal pole_decay_1
        pole_decay_1 = value
        update_controller()

    def update_decay_2(value: float) -> None:
        nonlocal pole_decay_2
        pole_decay_2 = value
        update_controller()

    def update_frequency(value: float) -> None:
        nonlocal pole_frequency
        pole_frequency = value
        update_controller()

    def update_pole_mode(label: str) -> None:
        nonlocal pole_mode
        pole_mode = label
        is_distinct_real_mode = pole_mode == "Two distinct real poles"
        is_conjugate_mode = pole_mode == "Conjugate pair"
        decay_2_slider.set_active(is_distinct_real_mode)
        frequency_slider.set_active(is_conjugate_mode)
        decay_2_slider_ax.set_visible(is_distinct_real_mode)
        frequency_slider_ax.set_visible(is_conjugate_mode)
        update_controller()

    setpoint_slider.on_changed(update_setpoint)
    decay_1_slider.on_changed(update_decay_1)
    decay_2_slider.on_changed(update_decay_2)
    frequency_slider.on_changed(update_frequency)
    pole_mode_selector.on_clicked(update_pole_mode)
    update_pole_mode(pole_mode)

    animation = FuncAnimation(
        figure,
        update,
        frames=frames,
        interval=20,
        blit=False,
        repeat=False,
    )
    _ = animation
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    plt.show()


if __name__ == "__main__":
    animate(PendulumConfig())
