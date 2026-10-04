# Before running this script, make sure to install the required libraries:
# pip install numpy matplotlib
#
# This script runs as a standalone Python file with matplotlib sliders.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider

# Pendulum dynamics
def pendulum_dynamics(theta, omega, g, L, m, b, dt):
    theta_dot = omega
    omega_dot = -(g/L) * np.sin(theta) - (b/(m*L**2)) * omega
    theta_new = theta + theta_dot * dt
    omega_new = omega + omega_dot * dt
    return theta_new, omega_new

# Simulation + animation function with sliders
def run_simulation():
    # Initial parameters
    theta0 = 0.5
    L = 1.0
    m = 1.0
    b = 0.1
    g = 9.81
    dt = 0.001
    steps = 50000

    def compute_trajectory(theta0, L, m, b, g, dt, steps):
        theta = theta0
        omega = 0.0
        thetas = []
        for _ in range(steps):
            theta, omega = pendulum_dynamics(theta, omega, g, L, m, b, dt)
            thetas.append(theta)
        x = L * np.sin(thetas)
        y = -L * np.cos(thetas)
        return x, y

    x, y = compute_trajectory(theta0, L, m, b, g, dt, steps)

    fig, ax = plt.subplots(figsize=(5, 5))
    plt.subplots_adjust(left=0.25, bottom=0.55)
    ax.set_xlim(-L-0.5, L+0.5)
    ax.set_ylim(-L-0.5, L+0.5)
    ax.set_aspect('equal')
    line, = ax.plot([], [], 'o-', lw=2)

    def update(frame):
        line.set_data([0, x[frame]], [0, y[frame]])
        return line,

    ani = FuncAnimation(fig, update, frames=len(x), interval=dt*1000, blit=True, repeat=False)

    # Slider axes
    axcolor = 'lightgoldenrodyellow'
    ax_theta0 = plt.axes([0.25, 0.45, 0.65, 0.03], facecolor=axcolor)
    ax_L = plt.axes([0.25, 0.40, 0.65, 0.03], facecolor=axcolor)
    ax_b = plt.axes([0.25, 0.35, 0.65, 0.03], facecolor=axcolor)
    ax_g = plt.axes([0.25, 0.30, 0.65, 0.03], facecolor=axcolor)
    ax_m = plt.axes([0.25, 0.25, 0.65, 0.03], facecolor=axcolor)
    ax_dt = plt.axes([0.25, 0.20, 0.65, 0.03], facecolor=axcolor)

    s_theta0 = Slider(ax_theta0, 'Theta0', -np.pi, np.pi, valinit=theta0)
    s_L = Slider(ax_L, 'Length', 0.2, 5.0, valinit=L)
    s_b = Slider(ax_b, 'Damping', 0.0, 2.0, valinit=b)
    s_g = Slider(ax_g, 'Gravity', 0.0, 20.0, valinit=g)
    s_m = Slider(ax_m, 'Mass', 0.1, 5.0, valinit=m)
    s_dt = Slider(ax_dt, 'dt', 0.005, 0.1, valinit=dt)

    def update_sliders(val):
        nonlocal theta0, L, m, b, g, dt, steps, x, y
        theta0 = s_theta0.val
        L = s_L.val
        b = s_b.val
        g = s_g.val
        m = s_m.val
        dt = s_dt.val

        x, y = compute_trajectory(theta0, L, m, b, g, dt, steps)
        ax.set_xlim(-L-0.5, L+0.5)
        ax.set_ylim(-L-0.5, L+0.5)
        fig.canvas.draw_idle()

    s_theta0.on_changed(update_sliders)
    s_L.on_changed(update_sliders)
    s_b.on_changed(update_sliders)
    s_g.on_changed(update_sliders)
    s_m.on_changed(update_sliders)
    s_dt.on_changed(update_sliders)

    plt.show()

if __name__ == "__main__":
    run_simulation()