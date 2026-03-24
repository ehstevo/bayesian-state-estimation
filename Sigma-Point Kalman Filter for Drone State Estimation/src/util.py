import numpy as np
import matplotlib.pyplot as plt

T = 0.1
Qd = np.diag([0.1**2, 0.1**2, 0.1**2,
        (2*np.pi/180)**2, (2*np.pi/180)**2])*T
R = np.diag([3.0**2, 3.0**2])
mu = np.array([0.0, 0, 25, 0, 0])
P0 = np.diag([10**2, 10**2, 3.0**2,
        (1.0/18)**2, (1.0/18)**2])


def cholup(R, v, w):
    n = np.size(v)
    for i in range(n):
        r = np.sqrt(R[i, i]**2 + w*v[i]**2)
        c = r/R[i, i]
        s = v[i]/R[i, i]
        R[i, i] = r
        R[i+1:n, i] = (R[i+1:n, i] + w*s*v[i+1:n])/c
        v[i+1:n]= c*v[i+1:n] - s*R[i+1:n, i]
    return R


def gen(K, T):
    # Constants
    RQd = np.linalg.cholesky(Qd)
    RR = np.linalg.cholesky(R)
    H = np.array([
        [1.0, 0, 0, 0, 0],
        [0, 1.0, 0, 0, 0]])
    g = 9.8     # acceleration of gravity
    qa = 0.05   # acceleration variance
    qw = 0.5    # roll rate variance
    ta = 1.0    # acceleration time constant
    tw = 10.0   # roll rate time constant

    # Get the FOGM coefficients.
    aa = np.exp(-T/ta)
    ba = qa*np.sqrt(1 - np.exp(-2*T/ta))
    aw = np.exp(-T/tw)
    bw = qw*np.sqrt(1 - np.exp(-2*T/tw))

    # Initialize the states.
    x = mu.copy()
    ua = qa * np.random.randn() # acceleration input
    iw = qw * np.random.randn() # roll rate FOGM state

    # Define the noise arrays.
    nu_t = np.random.randn(5, K)
    eta_t = np.random.randn(2, K)
    om_t = np.random.randn(2, K)

    # Allocate memory.
    Dx = np.zeros(5) # state derivatives
    x_t = np.zeros((5, K))
    u_t = np.zeros((2, K))
    z_t = np.zeros((2, K))

    for k in range(K):
        # Get the measurement.
        z = H @ x + RR @ eta_t[:, k]

        # Get the new inputs.
        ua = aa * ua + ba * om_t[0, k]
        iw = aw * iw + bw * om_t[1, k]
        uw = (iw - x[3])/T # roll rate input
        # This make the state x[3] equal to iw,
        # which will pull back to zero.

        # Storage
        x_t[:, k] = x
        u_t[0, k] = ua
        u_t[1, k] = uw
        z_t[:, k] = z

        # Get the derivatives.
        Dx[0] = x[2]*np.cos(x[4])
        Dx[1] = x[2]*np.sin(x[4])
        Dx[2] = ua
        Dx[3] = uw
        Dx[4] = g/x[2]*np.sin(x[3])

        # Integrate.
        x += Dx*T + RQd @ nu_t[:, k]

    return x_t, u_t, z_t


def plot_results(x_t, xh_t, Ph_t, T):

    # Build time.
    K = x_t.shape[1]
    t = np.arange(K)*T

    # Plot path.
    plt.figure()
    plt.plot(x_t[0], x_t[1], label="truth")
    plt.plot(xh_t[0], xh_t[1], label="estimate")
    plt.xlabel("Position, $p_x$ (m)")
    plt.ylabel("Position, $p_y$ (m)")
    plt.axis("equal")
    plt.legend()
    plt.savefig("fig_paths.pdf")

    # Extract the elements of the state covariance matrix.
    sx = np.sqrt(Ph_t[:, 0, 0])
    sy = np.sqrt(Ph_t[:, 1, 1])
    ss = np.sqrt(Ph_t[:, 2, 2])
    sr = np.sqrt(Ph_t[:, 3, 3])
    sh = np.sqrt(Ph_t[:, 4, 4])

    # Position errors
    plt.figure()
    plt.fill_between(t, -3*sx, 3*sx,
            alpha=0.3, color="tab:blue")
    plt.fill_between(t, -3*sy, 3*sy,
            alpha=0.3, color="tab:orange")
    plt.plot(t, x_t[0] - xh_t[0], color="tab:blue", label="$p_x$")
    plt.plot(t, x_t[1] - xh_t[1], color="tab:orange", label="$p_y$")
    plt.xlabel("Time, $t$ (s)")
    plt.ylabel("Position error (m)")
    plt.legend()
    plt.savefig(f"fig_position_errors.pdf")

    # Speed error
    plt.figure()
    plt.fill_between(t, -3*ss, 3*ss,
            alpha=0.3, color="tab:blue")
    plt.plot(t, x_t[2] - xh_t[2], color="tab:blue")
    plt.xlabel("Time, $t$ (s)")
    plt.ylabel("Speed error (m/s)")
    plt.savefig(f"fig_speed_error.pdf")

    # Angle errors
    plt.figure()
    plt.fill_between(t, -3*sr, 3*sr,
            alpha=0.3, color="tab:blue")
    plt.fill_between(t, -3*sh, 3*sh,
            alpha=0.3, color="tab:orange")
    plt.plot(t, x_t[3] - xh_t[3], color="tab:blue", label="roll")
    plt.plot(t, x_t[4] - xh_t[4], color="tab:orange", label="yaw")
    plt.xlabel("Time, $t$ (s)")
    plt.ylabel("Angle error (rad)")
    plt.legend()
    plt.savefig(f"fig_angle_errors.pdf")
    plt.show()