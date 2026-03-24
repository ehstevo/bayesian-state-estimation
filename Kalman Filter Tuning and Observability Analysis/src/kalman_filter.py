import numpy as np
import scipy.linalg as la
import util

# constants
T = 0.1                                 # sampling period (s)
T_dur = 100                             # simulation duration (s)
theta = np.pi/4                         # rotation of measurement sensor about the origin (rad)
# gamma = (4*np.pi)/180
K = int(T_dur/T)                        # total time steps in simulation
N = 4                                   # number of states
M = 2                                   # length of measurement vector

# system definition
F = np.zeros((4, 4))                    # State transition matrix (dynamics)
# F[0, 1] = gamma
F[0, 2] = 1
# F[1, 0] = -1*gamma
F[1, 3] = 1
# F[2, 3] = gamma
# F[3, 2] = -1*gamma

B = np.zeros((4, 2))                    # Control input matrix
B[2, 0] = 1
B[3, 1] = 1

H = np.zeros((2, 4))                    # Observation matrix (rotation from state's coordinate frame to sensor's)
H[0, 0] = np.cos(theta)
H[0, 1] = np.sin(theta)
H[1, 0] = np.sin(theta)*-1
H[1, 1] = np.cos(theta)
# H[1, 2] = 1

Q = np.diag([0.01, 0.01, 0.01, 0.01])   # Process noise covariance matrix
R = np.diag([1.0, 0.1])                 # Measurement noise covariance matrix

Phi, Bd, Qd = util.vanloan(F, B, Q, T)  # vanloan method to discretize F, B, and Q into Phi, Bd, and Qd

# load data
data = np.fromfile("data/xuz.bin").reshape((-1, 8)).T
x_t = data[:4]
u_t = data[4:6]
z_t = data[6:]

# scaling factors
s = [1/16, 1/9, 1/4, 1, 4, 9, 16]

# estimate using Kalman Filter
def estimate(x_t, u_t, z_t, Phi, Bd, Qd, H, R, T_dur):
    px = x_t[:,0:1]                         # initial state estimate
    P = np.eye(4)                           # initial state covariance matrix
    # P = P * s[i]                          # when scaling Q, must also scale P0

    # storage (for assignment 5 metrics)
    xh_t = np.zeros((4, T_dur))                 # state estimates
    xe_t = np.zeros((4, T_dur))                 # state errors (position)
    P_t = np.zeros((4, T_dur))                  # state covariance matrices

    # storage (for assignment 6 metrics)
    mse = np.zeros(T_dur)                       # stores the e_k^T * e_k term for each k
    anees = np.zeros(T_dur)                     # stores the e_k^T * P^-1 * e_k term for each k
    anis = np.zeros(T_dur)                      # stores the r^T * S^-1 * r term for each k

    for k in range(T_dur):
        # update
        S = (H @ P @ H.T) + R
        K = P @ H.T @ la.inv(S)                 # Kalman gain - weighting factor that balances predicted states's uncertainty
                                                # against the measurement's uncertainty
        r = z_t[:,k:k+1] - (H @ px)             # residual
        px = px + (K @ r)                       # new estimate = old estimate + gain * residual
        P = P - (K @ H @ P)                     # update state covariance matrix

        # if k==999:
            # print(np.round(P, 3))

        # store
        xh_t[:,k:k+1] = px                      # state estimate
        e_k = x_t[:,k:k+1] - px                 # state error
        xe_t[:,k:k+1] = e_k
        P_t[:,k:k+1] = np.diag(P).reshape(4, 1) # state covariance

        # store (MSE, ANEES, ANIS)
        mse[k] = (e_k.T @ e_k)[0][0]               # mean squared error
        anees[k] = (e_k.T @ la.inv(P) @ e_k)[0][0] # average normalized error estimate squared
        anis[k] = (r.T @ la.inv(S) @ r )[0][0]     # average normalized innovation squared

        # propagate
        px = (Phi @ px) + (Bd @ u_t[:,k:k+1])   # propagate estimate based on dynamics and control inputs
        P = (Phi @ P @ Phi.T) + Qd              # propagate state covariance matrix

    util.plot_paths(x_t, xh_t, z_t)
    util.plot_errors(xe_t, P_t)

    return xh_t, xe_t, P_t, mse, anees, anis

# calculates MSE, ANEES, and ANIS for different values of Q and R
def calculate_metrics():
    # store each combination of Q and R in 7x7 matrices
    mse_s = np.zeros((7, 7))
    anees_s = np.zeros((7, 7))
    anis_s = np.zeros((7, 7))

    for i in range(len(s)):
        for j in range(len(s)):
            Q = np.diag([0.01, 0.01, 0.01, 0.01]) * s[i]
            R = np.diag([1.0, 0.1]) * s[j]
            Phi, Bd, Qd = util.vanloan(F, B, Q, T) 

            _, _, P_t, mse, anees, anis = estimate(x_t, u_t, z_t, Phi, Bd, Qd, H, R, K)
            # print (mse[1], anees[1], anis[1])
            mse = np.sum(mse) / K
            anees = np.sum(anees) / (K*N)
            anis = np.sum(anis) / (K*M)

            mse_s[i, j] = mse
            anees_s[i, j] = anees
            anis_s[i, j] = anis

    util.plot_results(mse_s)
    util.plot_results(anees_s)
    util.plot_results(anis_s)

def calculate_observability(H, F, M, N):
    l = int(np.ceil(N/M))
    O = H
    for i in range(1, l):
        O = np.vstack((O, H @ F**i))
    
    U, s, Vt = np.linalg.svd(O)

    return O, s, Vt

# O, s, Vt = calculate_observability(H, F, M, N)
# print(O)
# print(s)
# print(Vt)
# print(Vt.T)

# calculate_metrics()

estimate(x_t, u_t, z_t, Phi, Bd, Qd, H, R, K)