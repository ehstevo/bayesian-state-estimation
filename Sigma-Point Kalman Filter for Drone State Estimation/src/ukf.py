import numpy as np
import matplotlib.pyplot as plt
import util
import time
from tqdm import tqdm

# constants
T = 0.1                                 # sampling period (s)
T_dur = 100                             # simulation duration (s)
g = 9.8                                 # acceleration of gravity (m/s)
np.random.seed(0)

Qd = util.Qd.copy()                     # process noise covariance
R = util.R.copy()                       # measurement noise covariance

# initial conditions
xh = util.mu.copy().reshape((-1, 1))    # initial state
Ph = util.P0.copy()                     # initial covariance

# square roots of covariance matrices
RP = np.linalg.cholesky(Ph)
RQd = np.linalg.cholesky(Qd)
RR = np.linalg.cholesky(R)

# sigma point transform
def spt(x, RP):
    # x - state
    # RP - state covariance
    n = x.shape[0]
    sig = np.zeros((n, 2*n)) + x
    sig[:,:n] += RP
    sig[:,n:] -= RP
    return sig

# h function to get predicted measurement
def h(x):
    return x[:2]

# phi function to propagate state through dynamics
def phi(x_in, u):
    x = x_in.copy()
    x[0] += x[2]*np.cos(x[4])*T
    x[1] += x[2]*np.sin(x[4])*T
    x[2] += u[0]*T
    x[3] += u[1]*T
    x[4] += (g/x[2])*np.sin(x[3])*T
    return x

# generate truth state, controls, and measurements
x_t, u_t, z_t = util.gen(int(T_dur/T), T)

def estimate(u_t, z_t, xh, RP, RQd, RR):
    n = len(xh)                         # number of states
    m = z_t.shape[0]                    # number of measurements
    Ts = z_t.shape[1]                   # number of steps in the simulation

    # Unscented parameters
    alpha = 1e-3
    beta = 2.0
    kappa = 0.0
    lamb = alpha**2 * (n + kappa) - n
    gamma = np.sqrt(n + lamb)
    w = (1.0/(2*(n + lamb)) * np.ones(2*n)).reshape((-1,1)) # sigma point weights for remaining n*2 terms
    wm0 = lamb/(n + lamb)                                   # sigma point weight 0 for mean
    wc0 = lamb/(n + lamb) + (1 - alpha**2 + beta)           # sigma point weight 0 for covariance

    # storage
    xh_t = np.zeros((n, Ts))
    Ph_t = np.zeros((Ts, n, n))

    for t in tqdm(range(int(T_dur/T))):
        # update
        sigma_points = spt(xh, RP*gamma)            # generate n*2 sigma points (xh +- RP)
        
        z = z_t[:,t:t+1]                            # get measurement
        Z0 = h(xh)                                  # get predicted measurement for mean (first sigma point)
        Z = h(sigma_points)                         # get predicted measurement for rest of sigma points
        zh = wm0*Z0 + (Z @ w)                       # calculate predicted measurement as weighted sum of sigma points
        
        diff = Z - zh                               # calculate the spread of possible measurements
        A = np.sqrt(w[:,0]) * diff                  # scale the spread matrix by the weights
        QR_up = np.hstack((A, RR))                  # build the matrix for qr decomposition, adding the cholesky R matrix side by side to the spread matrix
        S_y = np.linalg.qr(QR_up.T, mode="r")       # perform qr decomposition. This is a numerically stable way to perform "addition" 
                                                    # of uncertainties directly of their square-root factors. This combines our uncertainty from our state
                                                    # with our uncertainty from our measurement using the square-root factors.
        S_y = util.cholup(S_y.T, (Z0-zh)[:,0], wc0) # Now we have the full square-root of the innovation covariance, S_y, telling us how uncertain our measurements are
                                                    # we must do the cholup for the central sigma point because it may be negative
                                                    # we use S.T because qr returns an upper triangular matrix and cholup expects a lower triangular

        wc = np.vstack((np.array([[wc0]]), w))      # stacked covariance weights
        wc = np.diag(wc[:,0])                       # create a big diagonal for matrix multiplication
        xdev = np.hstack((xh-xh, sigma_points-xh))  # stack mean state with rest of sigma points
        ydev = np.hstack((Z0-zh, diff))             # stack mean measurement with rest of measurement spreads
        P_xy = xdev @ wc @ ydev.T                   # cross covariance, P_xy, measures how much the state moves if the measurement residual moves

        B = P_xy @ np.linalg.inv(S_y.T)
        K = B @ np.linalg.inv(S_y)                  # Kalman gain computed using the square root matrix of P_yy. 
                                                    # Tells us how much the state should move when the measurement residual changes

        xh = xh + K @ (z - zh)                      # state update
        U = K @ S_y                                 
        for j in range(U.shape[1]):                 # we need to loop through each column of U for the chol downdate
            RP = util.cholup(RP, U[:,j], -1)        # state covariance update. chol downdate removes the uncertainty UU^T)
                                            
        # store
        xh_t[:,t:t+1] = xh
        Ph_t[t,:,:] = RP @ RP.T

        # propagate
        sigma_points = spt(xh, RP*gamma)            # generate n*2 sigma points again

        X0 = phi(xh, u_t[:,t])                      # propagate mean through dynamics
        X = phi(sigma_points, u_t[:,t])             # propagate sigma points through dynamics

        wm = np.vstack((np.array([[wm0]]), w))      # stacked mean weights
        sig = np.hstack((X0, X))                    # stacked sigma points with mean sigma point
        xh = sig @ wm                               # predicted state mean, weighted sum of propagated sigma points

        diff_prop = np.sqrt(w[:,0])*(X-xh)          # covariance weighted sum of sigma points minus mean
        QR_prop = np.hstack((diff_prop, RQd))       # build the matrix for qr decomposition, adding the cholesky Q matrix side by side to the spread matrix
        S = np.linalg.qr(QR_prop.T, mode="r")       # perform QR decomposition again. combine dynamics and process noise.
        S = util.cholup(S.T, (X0-xh)[:,0], wc0)     # now we have the full square-root predicted state covariance, telling us how uncertain our predicted state is
                                                    # again we have to do the cholup because wc0 may be negative. We do S.T again because it expects a lower triangular
        RP = S

    return xh_t, Ph_t


# tic = time.perf_counter()
xh_t, Ph_t = estimate(u_t, z_t, xh, RP, RQd, RR)
# toc = time.perf_counter()
# print("Time elapsed:", toc - tic)
# error = (xh_t-x_t)**2
# error = np.mean(error)
# print(error)

util.plot_results(x_t, xh_t, Ph_t, T)