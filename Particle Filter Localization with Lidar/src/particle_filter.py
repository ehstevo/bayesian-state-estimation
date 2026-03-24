import numpy as np
import matplotlib.pyplot as plt
import util

# Read in truth, inputs, and measurement data
M = np.fromfile('data/xyaswr.bin').reshape((-1, 6)).T

# constants 
K = len(M[0])               # time steps
T = 0.1                     # sampling period (s)
Trt = np.sqrt(T)            # square root of sampling period
J = 650                     # number of particles
I = 2                       # number of input noise sources (control inputs)

# storage containers
xh_t = np.zeros((3, K))     # state estimates over time
vh_t = np.zeros((3, K))     # variance estimates over time

# separate data into truth, inputs, and measurements
x_t = M[:3,]                # truth data, x and y position (cm), angle a (rad)
u_t = M[3:5,]               # inputs, speed s (cm/s) and rotation rate w (rad/s)
z_t = M[-1:,]               # measurements, lidar range r (cm)

sigma_x = 100               # initial uncertainty of x position (cm)
sigma_y = 100               # initial uncertainty of y position (cm)
sigma_a = np.pi             # initial uncertainty of heading angle (rad)
sigma_s = 32                # noise in forward speed (cm/s)
sigma_w = 0.55              # noise in rotation rate (rad/s)
sigma_r = 10                # lidar measurement noise (cm)

P0 = np.diag([sigma_x**2, sigma_y**2, sigma_a**2])          # initial state uncertainty
Q = np.diag([sigma_s**2, sigma_w**2])                       # process noise (motion uncertainty)
R = sigma_r**2                                              # measurement noise

mu = x_t[:,0]                                               # initial truth state, center of our pdf we sample from to make particles

# state initialization
Prt = np.linalg.cholesky(P0)                                # cholesky decomposition gives us a "square root" matrix
x_j = mu[:,np.newaxis] + Prt @ np.random.randn(3, J)        # sample J particles from our initial pdf with mean = mu and std = Prt
Qrt = np.linalg.cholesky(Q)                                 # cholesky decomposition of Q

w = np.ones(J)/J                                            # initial weights

# estimation loop
for k in range(K):
    # update
    zh = util.h(x_j, k)                         # predicted measurements for each particle
    z = z_t[:,k]                                # actual measurement
    r = z - zh                                  # residuals for each particle
    L = w * np.exp((-0.5)*(r**2 / R))           # likelihood for each particle
    w = L / np.sum(L)                           # normalize the set, becomes weight update

    # estimate
    xh = x_j @ w                                # estimate the true state by computing a weighted average of particles

    d = xh[:, np.newaxis] - x_j                 # sum of differences between estimate and particle state
    Ph = (w*d) @ d.T                            # estimate of covariance matrix for moment in time

    # store
    xh_t[:,k] = xh
    vh_t[:,k] = np.diag(Ph)                     # stores diagonal of Ph (state variances)

    # resample
    Je = 1.0 / np.sum(w**2)                     # effective number of samples

    if Je < (0.6 * J):                                      # decide if time to resample
        comb = (np.random.rand() + np.arange(J))/J
        je = np.searchsorted(np.cumsum(w), comb)
        x_j = x_j[:, je]                                    # does not create new states, duplicates highest weighted states and discards lowest weighted states (in theory)
        w = np.ones(J) / J

    # propagate
    nu_j = Trt * (Qrt @ np.random.randn(I, J))              # select random values for control inputs white noise based on covariance matrix Q
    v = u_t[:,k].reshape(2, 1) + nu_j                       # control inputs + noise
    x_j[0,:] = x_j[0,:] + v[0,:]*np.cos(x_j[2])*T           # x position dynamics
    x_j[1,:] = x_j[1,:] + v[0,:]*np.sin(x_j[2])*T           # y position dynamics
    x_j[2,:] = x_j[2,:] + v[1,:]*T                          # a rotation dynamics


util.plot_results(x_t, xh_t, vh_t, z_t, T)




def plot_initial_particles(X, mu):
    J = X.shape[1]
    idx = np.arange(J)

    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(8, 6))

    labels = ['x (cm)', 'y (cm)', 'a (rad)']

    for i in range(3):
        axs[i].plot(idx, X[i,:], '.', markersize=6)
        axs[i].set_ylabel(labels[i])
        axs[i].grid(True)

        mu_i = mu.flatten()[i]
        axs[i].axhline(mu_i, color='r', linestyle='--')

    axs[-1].set_xlabel('particle index')

    fig.suptitle('Initial Particle States')
    plt.tight_layout()
    plt.show()

# plot_initial_particles(x_j, mu)

