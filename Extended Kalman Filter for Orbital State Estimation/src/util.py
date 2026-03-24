import numpy as np
import matplotlib.pyplot as plt


def gen(K, T):
    # Set the rng seed.
    np.random.seed(4)

    # Constants
    R = (0.2*np.pi/180)**2
    Q = np.diag([0.1**2, 0.1**2, 0.05**2, 0.05**2])
    GE = 3.986e14
    tau = 2*np.pi

    # Initialize the state.
    P = np.diag([1000.0**2, 1000**2, 100**2, 100**2])
    mu = np.array([2e7, 0.0, 0, 4500])
    x = mu + np.linalg.cholesky(P) @ np.random.randn(4)

    # Initialize the derivatives.
    DpxO = x[2]
    DpyO = x[3]
    r = np.sqrt(x[0]**2 + x[1]**2)
    DvxO = (-GE/r**3)*x[0]
    DvyO = (-GE/r**3)*x[1]

    # Allocate memory.
    nu_t = np.linalg.cholesky(Q*T) @ np.random.randn(4, K)
    et_t = np.sqrt(R)*np.random.randn(K)
    x_t = np.zeros((len(x), K))
    z_t = np.zeros(K)

    for k in range(K):
        # Measurement
        z = np.arctan2(x[1], x[0]) + et_t[k]

        # Storage
        x_t[:, k] = x
        z_t[k] = z

        # Derivatives of states
        r3 = np.sqrt(x[0]**2 + x[1]**2)**3
        Dpx = x[2]
        Dpy = x[3]
        Dvx = (-GE/r3)*x[0]
        Dvy = (-GE/r3)*x[1]

        # Integration
        x[0] += (Dpx + DpxO)/2*T + nu_t[0, k]
        x[1] += (Dpy + DpyO)/2*T + nu_t[1, k]
        x[2] += (Dvx + DvxO)/2*T + nu_t[2, k]
        x[3] += (Dvy + DvyO)/2*T + nu_t[3, k]

        # Old derivatives
        DpxO = Dpx
        DpyO = Dpy
        DvxO = Dvx
        DvyO = Dvy

    return x_t, z_t


def plot_results(t, x_t, xh_t, Ph_t):
    plt.figure()
    plt.plot(x_t[0]/1e6, x_t[1]/1e6, label='truth')
    plt.plot(xh_t[0]/1e6, xh_t[1]/1e6, label='estimate')
    plt.axis('equal')
    plt.xlabel('$x$-axis position (Mm)')
    plt.ylabel('$y$-axis position (Mm)')
    plt.legend()
    plt.savefig('fig_path.pdf')

    plt.figure()
    sx = np.sqrt(Ph_t[:, 0, 0])/1e3
    sy = np.sqrt(Ph_t[:, 1, 1])/1e3
    plt.fill_between(t/3600, 3*sx, -3*sx, alpha=0.2, color='tab:blue')
    plt.fill_between(t/3600, 3*sy, -3*sy, alpha=0.2, color='tab:orange')
    plt.plot(t/3600, (x_t[0] - xh_t[0])/1e3, label='$x$ axis')
    plt.plot(t/3600, (x_t[1] - xh_t[1])/1e3, label='$y$ axis')
    plt.xlabel('Time, $t$ (hrs)')
    plt.ylabel('Position error (km)')
    plt.legend()
    plt.savefig('fig_pos_errors.pdf')

    plt.figure()
    sx = np.sqrt(Ph_t[:, 2, 2])
    sy = np.sqrt(Ph_t[:, 3, 3])
    plt.fill_between(t/3600, 3*sx, -3*sx, alpha=0.2, color='tab:blue')
    plt.fill_between(t/3600, 3*sy, -3*sy, alpha=0.2, color='tab:orange')
    plt.plot(t/3600, x_t[2] - xh_t[2], label='$x$ axis')
    plt.plot(t/3600, x_t[3] - xh_t[3], label='$y$ axis')
    plt.xlabel('Time, $t$ (hrs)')
    plt.ylabel('Velocity error (m/s)')
    plt.legend()
    plt.savefig('fig_vel_errors.pdf')
    plt.show()