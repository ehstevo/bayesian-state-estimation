import numpy as np
import scipy.linalg as la
import util

# constants
T = 0.1                                 # sampling period (s)
T_dur = 100                             # simulation duration (s)
theta = np.pi/4                         # rotation of measurement sensor about the origin (rad)
K = int(T_dur/T)                        # total time steps in simulation

# system definition
F = np.zeros((4, 4))                    # State transition matrix (dynamics)
F[0, 2] = 1
F[1, 3] = 1

B = np.zeros((4, 2))                    # Control input matrix
B[2, 0] = 1
B[3, 1] = 1

H = np.zeros((2, 4))                    # Observation matrix (rotation from state's coordinate frame to sensor's)
H[0, 0] = np.cos(theta)
H[0, 1] = np.sin(theta)
H[1, 0] = np.sin(theta)*-1
H[1, 1] = np.cos(theta)

Q = np.diag([0.01, 0.01, 0.01, 0.01])   # Process noise covariance matrix
R = np.diag([1.0, 0.1])                 # Measurement noise covariance matrix

Phi, Bd, Qd = util.vanloan(F, B, Q, T)  # vanloan method to discretize F, B, and Q into Phi, Bd, and Qd

# load data
data = np.fromfile("data/xuz.bin").reshape((-1, 8)).T
x_t = data[:4]
u_t = data[4:6]
z_t = data[6:]

px = x_t[:,0:1]                         # initial state estimate
P = np.eye(4)                           # initial state covariance matrix

# storage
xh_t = np.zeros((4, K))                 # state estimates
xe_t = np.zeros((4, K))                 # state errors (position)
P_t = np.zeros((4, K))                  # state covariance matrices

# estimate using Kalman Filter
for k in range(K):
    # update
    S = (H @ P @ H.T) + R
    K = P @ H.T @ la.inv(S)                 # Kalman gain - weighting factor that balances predicted states's uncertainty
                                            # against the measurement's uncertainty
    r = z_t[:,k:k+1] - (H @ px)             # residual
    px = px + (K @ r)                       # new estimate = old estimate + gain * residual
    P = P - (K @ H @ P)                     # update state covariance matrix

    if k==999:
        print(np.round(P, 3))

    # store
    xh_t[:,k:k+1] = px                      # state estimate
    xe_t[:,k:k+1] = x_t[:,k:k+1] - px       # state error
    P_t[:,k:k+1] = np.diag(P).reshape(4, 1)  # state covariance

    # propagate
    px = (Phi @ px) + (Bd @ u_t[:,k:k+1])   # propagate estimate based on dynamics and control inputs
    P = (Phi @ P @ Phi.T) + Qd              # propagate state covariance matrix


util.plot_paths(x_t, xh_t, z_t)
util.plot_errors(xe_t, P_t)