"""Animate a torque-controlled pendulum using continuous-time LQR.

Run with: python inverted_pendulum_lqr.py

Enter Q as a 2x2 matrix such as ``1, 0; 0, 1`` and R as a positive scalar
such as ``0.1`` in the simulation window, then press Enter. The angle is
measured from the downward vertical, so the downward position is 0 rad.
"""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, TextBox


@dataclass(frozen=True)
class PendulumConfig:
    mass: float = 0.5
    length: float = 1.0
    damping: float = 0.05
    gravity: float = 9.81
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
        if self.damping < 0:
            raise ValueError("Damping cannot be negative.")


def linearize_model(
    config: PendulumConfig, setpoint: float
) -> tuple[np.ndarray, np.ndarray]:
    """Linearize about an equilibrium at ``setpoint``."""
    inertia = config.mass * config.length**2
    state_matrix = np.array(
        (
            (0.0, 1.0),
            (
                -config.gravity / config.length * np.cos(setpoint),
                -config.damping / inertia,
            ),
        )
    )
    input_matrix = np.array(((0.0,), (1.0 / inertia,)))
    return state_matrix, input_matrix


def validate_weights(
    state_weight: np.ndarray, input_weight: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and return the LQR state and input weighting matrices."""
    q = np.asarray(state_weight, dtype=float)
    r = np.asarray(input_weight, dtype=float)
    if q.shape != (2, 2):
        raise ValueError("Q must be a 2x2 matrix.")
    if r.shape != (1, 1):
        raise ValueError("R must be a 1x1 matrix (a scalar).")
    if not np.isfinite(q).all() or not np.isfinite(r).all():
        raise ValueError("Q and R must contain only finite numbers.")
    if not np.allclose(q, q.T, atol=1e-10):
        raise ValueError("Q must be symmetric.")
    if not np.allclose(r, r.T, atol=1e-10):
        raise ValueError("R must be symmetric.")
    if np.linalg.eigvalsh(q).min() < -1e-10:
        raise ValueError("Q must be positive semidefinite.")
    if np.linalg.eigvalsh(r).min() <= 0:
        raise ValueError("R must be positive definite.")
    return q, r


def solve_continuous_riccati(
    state_matrix: np.ndarray,
    input_matrix: np.ndarray,
    state_weight: np.ndarray,
    input_weight: np.ndarray,
) -> np.ndarray:
    """Solve the continuous-time algebraic Riccati equation for this system."""
    q, r = validate_weights(state_weight, input_weight)
    n = state_matrix.shape[0]
    if state_matrix.shape != (n, n) or input_matrix.shape[0] != n:
        raise ValueError("Incompatible state-space matrix dimensions.")

    inverse_r = np.linalg.inv(r)
    hamiltonian = np.block(
        [
            [state_matrix, -input_matrix @ inverse_r @ input_matrix.T],
            [-q, -state_matrix.T],
        ]
    )
    eigenvalues, eigenvectors = np.linalg.eig(hamiltonian)
    stable = np.flatnonzero(eigenvalues.real < -1e-9)
    if stable.size != n:
        raise ValueError("The Riccati equation has no stabilizing solution.")

    stable_vectors = eigenvectors[:, stable]
    upper = stable_vectors[:n, :]
    lower = stable_vectors[n:, :]
    if np.linalg.cond(upper) > 1e12:
        raise ValueError("The Riccati solution is numerically ill-conditioned.")
    solution = np.linalg.solve(upper.T, lower.T).T
    if np.max(np.abs(solution.imag)) > 1e-8:
        raise ValueError("The Riccati solution is not real.")
    solution = solution.real
    solution = 0.5 * (solution + solution.T)
    residual = (
        state_matrix.T @ solution
        + solution @ state_matrix
        - solution @ input_matrix @ inverse_r @ input_matrix.T @ solution
        + q
    )
    if not np.allclose(residual, 0.0, atol=1e-7, rtol=1e-7):
        raise ValueError("Failed to compute an accurate Riccati solution.")
    return solution


def lqr_gain(
    config: PendulumConfig,
    setpoint: float,
    state_weight: np.ndarray,
    input_weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the stabilizing LQR gain and Riccati solution P."""
    state_matrix, input_matrix = linearize_model(config, setpoint)
    p_matrix = solve_continuous_riccati(
        state_matrix, input_matrix, state_weight, input_weight
    )
    gain = np.linalg.solve(
        input_weight,
        input_matrix.T @ p_matrix,
    )
    return gain, p_matrix


def _equilibrium_torque(config: PendulumConfig, setpoint: float) -> float:
    return (
        config.mass
        * config.length
        * config.gravity
        * np.sin(setpoint)
    )


def _control_torque(
    config: PendulumConfig,
    state: np.ndarray,
    setpoint: float,
    gain: np.ndarray,
) -> float:
    error = np.array((state[0] - setpoint, state[1]))
    torque = _equilibrium_torque(config, setpoint) - float(gain @ error)
    return float(np.clip(torque, -config.max_torque, config.max_torque))


def _state_derivative(
    config: PendulumConfig,
    state: np.ndarray,
    setpoint: float,
    gain: np.ndarray,
) -> np.ndarray:
    angle, angular_velocity = state
    inertia = config.mass * config.length**2
    angular_acceleration = (
        -config.gravity / config.length * np.sin(angle)
        + (
            _control_torque(config, state, setpoint, gain)
            - config.damping * angular_velocity
        )
        / inertia
    )
    return np.array((angular_velocity, angular_acceleration))


def _parse_matrix(value: str) -> np.ndarray:
    """Parse semicolon-separated matrix rows and comma-separated columns."""
    rows = [
        [float(element.strip()) for element in row.split(",")]
        for row in value.split(";")
    ]
    if not rows or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("Matrix rows must all have the same number of values.")
    return np.array(rows, dtype=float)


def animate(config: PendulumConfig) -> None:
    q_matrix = np.diag((10.0, 1.0))
    r_matrix = np.array(((0.5,),))
    current_setpoint = config.setpoint
    current_gain, current_p = lqr_gain(
        config, current_setpoint, q_matrix, r_matrix
    )

    step_count = int(np.ceil(config.duration / config.dt))
    times = np.linspace(0.0, config.duration, step_count + 1)
    dt = times[1] - times[0]
    states = np.zeros((step_count + 1, 2), dtype=float)
    states[0] = (config.initial_angle, config.initial_angular_velocity)
    torques = np.full(step_count + 1, np.nan)
    torques[0] = _control_torque(
        config, states[0], current_setpoint, current_gain
    )
    current_step = 0
    radius = config.length
    frames = list(range(0, len(times), max(1, int(round(0.02 / dt)))))
    if frames[-1] != len(times) - 1:
        frames.append(len(times) - 1)

    figure = plt.figure(figsize=(12, 10))
    layout = figure.add_gridspec(
        7,
        2,
        width_ratios=(1, 1.5),
        height_ratios=(1, 1, 1, 0.18, 0.22, 0.25, 0.35),
    )
    pendulum_ax = figure.add_subplot(layout[:3, 0])
    angle_ax = figure.add_subplot(layout[:2, 1])
    torque_ax = figure.add_subplot(layout[2, 1])
    setpoint_slider_ax = figure.add_subplot(layout[3, :])
    q_text_ax = figure.add_subplot(layout[4, 0])
    r_text_ax = figure.add_subplot(layout[4, 1])
    p_ax = figure.add_subplot(layout[5:, :])

    pendulum_ax.set(
        xlim=(-1.25 * radius, 1.25 * radius),
        ylim=(-1.25 * radius, 1.25 * radius),
        aspect="equal",
        title="Pendulum",
        xlabel="x [m]",
        ylabel="y [m]",
    )
    pendulum_ax.axhline(0, color="0.6", linewidth=1)
    pendulum_ax.plot(0, 0, "ko", markersize=7)
    target_rod, = pendulum_ax.plot(
        [0, radius * np.sin(current_setpoint)],
        [0, -radius * np.cos(current_setpoint)],
        "--",
        color="tab:green",
        linewidth=2,
        label="Desired position",
    )
    rod, = pendulum_ax.plot([], [], "o-", color="tab:blue", linewidth=4)
    pendulum_ax.legend(loc="lower left")

    angle_ax.set(
        xlim=(0, config.duration),
        ylim=(-180, 180),
        title="Angle response",
        xlabel="Time [s]",
        ylabel="Angle [deg]",
    )
    setpoint_line = angle_ax.axhline(
        np.rad2deg(current_setpoint),
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
        title="LQR control effort",
    )
    torque_ax.axhline(0, color="0.5", linewidth=0.8)
    torque_line, = torque_ax.plot([], [], color="tab:orange")
    torque_ax.grid(True, alpha=0.3)

    setpoint_slider = Slider(
        setpoint_slider_ax,
        "Desired position [deg]",
        -180.0,
        180.0,
        valinit=np.rad2deg(current_setpoint),
    )
    q_text_box = TextBox(
        q_text_ax, "Q (2x2)", initial="10, 0; 0, 1"
    )
    r_text_box = TextBox(r_text_ax, "R (scalar)", initial="0.5")
    p_ax.set_axis_off()
    p_text = p_ax.text(
        0.01,
        0.92,
        "",
        va="top",
        family="monospace",
        transform=p_ax.transAxes,
    )
    status_text = p_ax.text(
        0.55,
        0.92,
        "",
        va="top",
        color="tab:red",
        transform=p_ax.transAxes,
    )

    def render(frame: int) -> tuple:
        nonlocal current_step
        for index in range(current_step, frame):
            state = states[index]
            k1 = _state_derivative(config, state, current_setpoint, current_gain)
            k2 = _state_derivative(
                config, state + 0.5 * dt * k1, current_setpoint, current_gain
            )
            k3 = _state_derivative(
                config, state + 0.5 * dt * k2, current_setpoint, current_gain
            )
            k4 = _state_derivative(
                config, state + dt * k3, current_setpoint, current_gain
            )
            states[index + 1] = state + (dt / 6.0) * (
                k1 + 2 * k2 + 2 * k3 + k4
            )
            torques[index + 1] = _control_torque(
                config, states[index + 1], current_setpoint, current_gain
            )
        current_step = frame
        angle_degrees = np.rad2deg(states[: frame + 1, 0])
        angle = states[frame, 0]
        rod.set_data(
            [0, radius * np.sin(angle)],
            [0, -radius * np.cos(angle)],
        )
        angle_line.set_data(times[: frame + 1], angle_degrees)
        angle_marker.set_data([times[frame]], [angle_degrees[-1]])
        torque_line.set_data(times[: frame + 1], torques[: frame + 1])
        figure.suptitle(
            f"LQR control  |  time = {times[frame]:.2f} s  |  "
            f"K = [{current_gain[0, 0]:.2f}, {current_gain[0, 1]:.2f}]"
        )
        return rod, angle_line, angle_marker, torque_line

    def refresh_p() -> None:
        p_text.set_text(
            "Riccati solution P =\n"
            + np.array2string(current_p, precision=4, suppress_small=True)
            + f"\n\nGain K = {np.array2string(current_gain, precision=4)}"
        )

    def update_controller(
        q_text: str | None = None, r_text: str | None = None
    ) -> None:
        nonlocal current_gain, current_p
        try:
            q_candidate = _parse_matrix(
                q_text_box.text if q_text is None else q_text
            )
            r_candidate = _parse_matrix(
                r_text_box.text if r_text is None else r_text
            )
            gain, p_matrix = lqr_gain(
                config, current_setpoint, q_candidate, r_candidate
            )
        except (ValueError, np.linalg.LinAlgError) as error:
            status_text.set_text(str(error))
            figure.canvas.draw_idle()
            return
        current_gain, current_p = gain, p_matrix
        torques[current_step] = _control_torque(
            config, states[current_step], current_setpoint, current_gain
        )
        status_text.set_text("")
        refresh_p()
        render(current_step)
        figure.canvas.draw_idle()

    def update_setpoint(value: float) -> None:
        nonlocal current_setpoint
        current_setpoint = float(np.deg2rad(value))
        setpoint_line.set_ydata((value, value))
        target_rod.set_data(
            [0, radius * np.sin(current_setpoint)],
            [0, -radius * np.cos(current_setpoint)],
        )
        angle_ax.set_ylim(
            -max(180.0, abs(value) * 1.2),
            max(180.0, abs(value) * 1.2),
        )
        update_controller()

    setpoint_slider.on_changed(update_setpoint)
    q_text_box.on_submit(lambda value: update_controller(q_text=value))
    r_text_box.on_submit(lambda value: update_controller(r_text=value))
    refresh_p()
    animation = FuncAnimation(
        figure,
        render,
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
