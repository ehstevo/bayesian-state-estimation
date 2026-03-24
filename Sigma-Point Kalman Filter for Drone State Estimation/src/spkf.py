import numpy as np
import matplotlib.pyplot as plt
import util
import time
import tkz
from tqdm import tqdm


def spt(sig, x, R):
    n = len(x)
    sig[:, :] = x[:, None]
    sig[:, :n] += R
    sig[:, n:] -= R


def phi(xIn, u, T):
    x = xIn + 0
    x[0] += xIn[2]*np.cos(xIn[4])*T
    x[1] += xIn[2]*np.sin(xIn[4])*T
    x[2] += u[0]*T
    x[3] += u[1]*T
    x[4] += 9.8/xIn[2]*np.sin(xIn[3])*T
    return x


def h(x):
    return x[:2]


def ukf(u_t, z_t, xh, RP, RQd, RR, T):
    # Dimensions
    K = u_t.shape[1]
    n = len(xh)
    m = z_t.shape[0]

    # Tuning
    alpha = 0.001
    beta = 2.0
    kappa = 0.0
    lamb = alpha**2 * (n + kappa) - n
    gamma = np.sqrt(n + lamb)
    w = 1.0/(2*(n + lamb)) * np.ones(2*n)
    wm0 = lamb/(n + lamb)
    wc0 = lamb/(n + lamb) + (1 - alpha**2 + beta)

    # Allocate memory.
    sig = np.zeros((n, 2*n))
    xh_t = np.zeros((n, K))
    Ph_t = np.zeros((K, n, n))

    # Initialize the augmented matrices.
    Lu = np.zeros((2*n + m, m + n))
    Lu[2*n:, :m] = RR.T
    Lp = np.zeros((3*n, n))
    Lp[2*n:] = RQd.T

    # For each time sample,
    for k in tqdm(range(K)):
        # Parse the inputs and measurements.
        u = u_t[:, k]
        z = z_t[:, k]

        # Sigma-point transform.
        spt(sig, xh, gamma*RP)

        # Update.
        Z0 = h(xh)
        Z = h(sig)
        zh = wm0*Z0 + np.sum(w*Z, axis=1)
        Lu[:2*n, :m] = np.sqrt(w[:, None])*(Z.T - zh)
        Lu[:n, m:] = RP.T/np.sqrt(2)
        Lu[n:2*n, m:] = -RP.T/np.sqrt(2)
        U = np.linalg.qr(Lu, mode="r")
        RS = U[:m, :m].T
        W = U[:m, m:].T
        RP = U[m:(m + n), m:].T
        Kg = W @ np.linalg.inv(RS)
        xh += Kg @ (z - zh)

        # Store.
        xh_t[:, k] = xh
        Ph_t[k, :, :] = RP @ RP.T

        # Sigma-point transform.
        spt(sig, xh, gamma*RP)

        # Propagate.
        X0 = phi(xh, u, T)
        X = phi(sig, u, T)
        xh = wm0*X0 + np.sum(w*X, axis=1)
        Lp[:2*n] = np.sqrt(w[:, None])*(X.T - xh)
        RP = np.linalg.qr(Lp, mode="r").T
        RP = util.cholup(RP, X0 - xh, wc0)

    return xh_t, Ph_t


def cubature_normal(u_t, z_t, xh, Ph, Qd, R, T):
    # Dimensions
    K = u_t.shape[1]
    n = len(xh)
    m = z_t.shape[0]

    # Tuning
    gamma = np.sqrt(n)
    w = 1.0/(2*n)

    # Allocate memory.
    sig = np.zeros((n, 2*n))
    xh_t = np.zeros((n, K))
    Ph_t = np.zeros((K, n, n))

    # Initialize propagated sigma points.
    RP = np.linalg.cholesky(Ph)
    spt(sig, xh, gamma*RP)
    X = sig + 0

    # For each time sample,
    for k in range(K):
        # Parse the inputs and measurements.
        u = u_t[:, k]
        z = z_t[:, k]

        # Sigma-point transform.
        RP = np.linalg.cholesky(Ph)
        spt(sig, xh, gamma*RP)

        # Update.
        Z = h(sig)
        zh = w*np.sum(Z, axis=1)
        S = w*(Z.T - zh).T @ (Z.T - zh) + R
        Cxz = w*(X.T - xh).T @ (Z.T - zh)
        Kg = Cxz @ np.linalg.inv(S)
        xh += Kg @ (z - zh)
        Ph -= Kg @ S @ Kg.T

        # Store.
        xh_t[:, k] = xh
        Ph_t[k, :, :] = RP @ RP.T

        # Sigma-point transform.
        RP = np.linalg.cholesky(Ph)
        spt(sig, xh, gamma*RP)

        # Propagate.
        X = phi(sig, u, T)
        xh = w*np.sum(X, axis=1)
        Ph = w*(X.T - xh).T @ (X.T - xh) + Qd

    return xh_t, Ph_t


def cubature(u_t, z_t, xh, RP, RQd, RR, T):
    # Dimensions
    m, K = z_t.shape
    n = len(xh)

    # Tuning
    gamma = np.sqrt(n)
    w = 1.0/(2*gamma**2)

    # Allocate memory.
    sig = np.zeros((n, 2*n))
    xh_t = np.zeros((n, K))
    Ph_t = np.zeros((K, n, n))

    # Initialize the augmented matrices.
    Lu = np.zeros((2*n + m, m + n))
    Lu[2*n:, :m] = RR.T
    Lp = np.zeros((3*n, n))
    Lp[2*n:] = RQd.T

    # For each time sample,
    for k in range(K):
        # Parse the inputs and measurements.
        u = u_t[:, k]
        z = z_t[:, k]

        # Sigma-point transform.
        spt(sig, xh, gamma*RP)

        # Update.
        Z = h(sig)
        zh = w*np.sum(Z, axis=1)
        Lu[:2*n, :m] = np.sqrt(w)*(Z.T - zh)
        Lu[:n, m:] = RP.T/np.sqrt(2)
        Lu[n:2*n, m:] = -RP.T/np.sqrt(2)
        U = np.linalg.qr(Lu, mode="r")
        RS = U[:m, :m].T
        W = U[:m, m:].T
        RP = U[m:(n + m), m:].T
        Kg = W @ np.linalg.inv(RS)
        xh += Kg @ (z - zh)

        # Store.
        xh_t[:, k] = xh
        Ph_t[k, :, :] = RP @ RP.T

        # Sigma-point transform.
        spt(sig, xh, gamma*RP)

        # Propagate.
        X = phi(sig, u, T)
        xh = w*np.sum(X, axis=1)
        Lp[:2*n] = np.sqrt(w)*(X.T - xh)
        RP = np.linalg.qr(Lp, mode="r").T

    return xh_t, Ph_t


def taylor(u_t, z_t, xh, RP, RQd, RR, T):
    # Dimensions
    K = u_t.shape[1]
    n = len(xh)
    m = z_t.shape[0]

    # Tuning
    gamma = np.sqrt(3)
    w = 1.0/(2*gamma**2)

    # Allocate memory.
    sig = np.zeros((n, 2*n))
    xh_t = np.zeros((n, K))
    Ph_t = np.zeros((K, n, n))

    # Initialize the augmented matrices.
    Lu = np.zeros((2*n + m, m + n))
    Lu[2*n:, :m] = RR.T
    Lp = np.zeros((3*n, n))
    Lp[2*n:, :] = RQd.T

    # For each time sample,
    for k in range(K):
        # Parse the inputs and measurements.
        u = u_t[:, k]
        z = z_t[:, k]

        # Sigma-point transform.
        spt(sig, xh, gamma*RP)

        # Update.
        Z = h(sig)
        zh = h(xh)
        Lu[:2*n, :m] = np.sqrt(w)*(Z.T - zh)
        Lu[:n, m:] = RP.T/np.sqrt(2)
        Lu[n:2*n, m:] = -RP.T/np.sqrt(2)
        U = np.linalg.qr(Lu, mode="r")
        RS = U[:m, :m].T
        RP = U[m:(n + m), m:].T
        W = U[:m, m:].T
        Kg = W @ np.linalg.inv(RS)
        xh += Kg @ (z - zh)

        # Store.
        xh_t[:, k] = xh
        Ph_t[k, :, :] = RP @ RP.T

        # Sigma-point transform.
        spt(sig, xh, gamma*RP)

        # Propagate.
        X = phi(sig, u, T)
        xh = phi(xh, u, T)
        Lp[:2*n, :] = np.sqrt(w)*(X.T - xh)
        RP = np.linalg.qr(Lp, mode="r").T

    return xh_t, Ph_t


# Load and parse the data.
T = 0.1
K = 10000
np.random.seed(0)
x_t, u_t, z_t = util.gen(K, T)

# Constants
Qd = util.Qd.copy()
R = util.R.copy()

# Initial conditions
xh = util.mu.copy()
Ph = util.P0.copy()

# Square roots of covariance matrices.
RP = np.linalg.cholesky(Ph)
RQd = np.linalg.cholesky(Qd)
RR = np.linalg.cholesky(R)

# Sigma-point Kalman Filter.
tic = time.perf_counter()
xh_t, Ph_t = ukf(u_t, z_t, xh, RP, RQd, RR, T)
#xh_t, Ph_t = cubature_normal(u_t, z_t, xh, Ph, Qd, R, T)
#xh_t, Ph_t = cubature(u_t, z_t, xh, RP, RQd, RR, T)
#xh_t, Ph_t = taylor(u_t, z_t, xh, RP, RQd, RR, T)
toc = time.perf_counter()
print("Time elapsed:", toc - tic)

if 0:
    # Build time.
    t = np.arange(K)*T

    # Plot path.
    fig = tkz.graph(f"fig_paths")
    fig.plot(x_t[0]/1000, x_t[1]/1000, label="truth", color=0x1f77b4)
    fig.plot(xh_t[0]/1000, xh_t[1]/1000, label="estimate", color=0xff7f0e)
    fig.xlabel = "Position, $p_x$ (km)"
    fig.ylabel = "Position, $p_y$ (km)"
    fig.equal = True
    fig.render()

    # Extract the elements of the state covariance matrix.
    sx = np.sqrt(Ph_t[:, 0, 0])
    sy = np.sqrt(Ph_t[:, 1, 1])
    ss = np.sqrt(Ph_t[:, 2, 2])
    sr = np.sqrt(Ph_t[:, 3, 3])
    sh = np.sqrt(Ph_t[:, 4, 4])

    # Position errors
    fig = tkz.graph(f"fig_position_errors")
    fig.fill(t, -3*sx, 3*sx, color=0x1f77b4, opacity=0.3)
    fig.fill(t, -3*sy, 3*sy, color=0xff7f0e, opacity=0.3)
    fig.plot(t, x_t[0] - xh_t[0], color=0x1f77b4, label="$p_x$")
    fig.plot(t, x_t[1] - xh_t[1], color=0xff7f0e, label="$p_y$")
    fig.xlabel = "Time, $t$ (s)"
    fig.ylabel = "Position error (m)"
    fig.render()

    # Speed error
    fig = tkz.graph(f"fig_speed_error")
    fig.fill(t, -3*ss, 3*ss, color=0x1f77b4, opacity=0.3)
    fig.plot(t, x_t[2] - xh_t[2], color=0x1f77b4)
    fig.xlabel = "Time, $t$ (s)"
    fig.ylabel = "Speed error (m/s)"
    fig.render()

    # Position errors
    fig = tkz.graph(f"fig_angle_errors")
    fig.fill(t, -3*sr, 3*sr, color=0x1f77b4, opacity=0.3)
    fig.fill(t, -3*sh, 3*sh, color=0xff7f0e, opacity=0.3)
    fig.plot(t, x_t[3] - xh_t[3], color=0x1f77b4, label="roll")
    fig.plot(t, x_t[4] - xh_t[4], color=0xff7f0e, label="yaw")
    fig.xlabel = "Time, $t$ (s)"
    fig.ylabel = "Angle error (rad)"
    fig.render()
else:
    util.plot_results(x_t, xh_t, Ph_t, T)

plt.show()
