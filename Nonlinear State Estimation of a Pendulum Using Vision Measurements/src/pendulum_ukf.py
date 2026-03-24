import numpy as np
import matplotlib.pyplot as plt
import util
import time
from tqdm import tqdm
import animation

# constants
T = 0.01                                    # sampling period
g = np.array([0, 8.48704896, 4.9]).T
k_d = 0.1                                   # drag coefficient
r_b = 12*(0.0254) / (2*np.pi)               # true radius of the ball
k = 900.0                                   # camera scaling factor
c_x = 640                                   # horizontal center of the camera image in pixels
c_y = 360                                   # vertical center of the camera image in pixels

# ukf constants
n = 9                                       # number of states
m = 3                                       # number of measurements
alpha = 0.1                                 # controls spread of sigma points from the mean
beta = 2.0                                  # incorporates prior knowledge about the distribution of x
kappa = 0.0                                 # idk what this does

z_t = np.fromfile("data/z_t1.bin").reshape((-1, 3)).T
z_t2 = np.fromfile("data/z_t2.bin").reshape((-1, 3)).T
z_t3 = np.fromfile("data/z_t3.bin").reshape((-1, 3)).T

def calculate_accelerations(state):
    # calculate ag (counter-gravitational acceleration)
    s_x = state[0][0]-state[6][0]
    s_y = state[1][0]-state[7][0]
    s_z = state[2][0]-state[8][0]
    s = np.array([s_x, s_y, s_z])
    u_s = s/np.linalg.norm(s)
    ag = -u_s*(np.dot(u_s, g))

    # calculate ac (centripetal acceleration)
    v = np.array([state[3][0], state[4][0], state[5][0]])
    ac = -u_s*(np.linalg.norm(v)**2/np.linalg.norm(s))

    # calculate ad (drag acceleration)
    ad = -v*(k_d*np.linalg.norm(v))

    return ag, ac, ad

def spt(x, RP):
    # x - state
    # RP - state covariance
    n = x.shape[0]
    sig = np.zeros((n, 2*n)) + x
    sig[:,:n] += RP
    sig[:,n:] -= RP
    return sig

def phi(sigma_points):
    phi = np.zeros_like(sigma_points)
    second = np.zeros((9,9))
    second[0][3] = 1
    second[1][4] = 1
    second[2][5] = 1
    second = second * T
    third = np.zeros((9,1))
    for i in range(sigma_points.shape[1]):
        x = sigma_points[:,i:i+1]
        ag, ac, ad = calculate_accelerations(x)
        a = ag + ac + ad + g
        third = np.zeros((9,1))
        third = np.vstack((0.5*(T**2)*a[0], 0.5*(T**2)*a[1], 0.5*(T**2)*a[2], T*a[0], T*a[1], T*a[2], 0, 0, 0))
        phi[:,i:i+1] = x + (second @ x) + third
    return phi

def h(sigma_points):
    rp_vec = k*(r_b / sigma_points[2,:])
    u_vec = c_x + k*(sigma_points[0,:]/sigma_points[2,:])
    v_vec = c_y + k*(sigma_points[1,:]/sigma_points[2,:])
    return np.vstack((rp_vec, u_vec, v_vec))

Qd = np.eye(9)*0.0001*T                     # Process noise covariance
R = np.eye(3)*0.001                         # Measurement noise covariance
RR = np.linalg.cholesky(R).T
RQd = np.linalg.cholesky(Qd).T

def estimate(z_t, T, T_dur):
    # initial state
    # derive initial position from first measurement and inverse of the measurement function
    # r_p is the radius of the ball in pixels, u and v are the horizontal and vertical pixel coordinates
    r_p0, u0, v0 = z_t[:,0]
    z0 = k*(r_b / r_p0)
    x0 = (u0-c_x)*(z0/k)
    y0 = (v0-c_y)*(z0/k)
    # derive initial velocity from rate of change of first measurement and second measurement
    r_p1, u1, v1 = z_t[:,1]
    z1 = k*(r_b / r_p1)
    x1 = (u1-c_x)*(z1/k)
    y1 = (v1-c_y)*(z1/k)
    v_x0 = (x1-x0) / T
    v_y0 = (y1-y0) / T
    v_z0 = (z1-z0) / T
    # define initial hook state, assume 1 meter from camera and 1 meter above ball
    h_x0 = 0
    h_y0 = -1
    h_z0 = 1
    xh = np.array([[x0], [y0], [z0], [v_x0], [v_y0], [v_z0], [h_x0], [h_y0], [h_z0]])
    # initial covariance
    P = np.eye(9)*0.1
    RP = np.linalg.cholesky(P)

    # storage
    xh_t = np.zeros((9, T_dur))                                 # estimates
    Ph_t = np.zeros((T_dur, 9, 9))                              # covariances
    iou_t = []

    # Unscented parameters
    alpha = 1e-3
    beta = 2.0
    kappa = 0.0
    lamb = alpha**2 * (n + kappa) - n
    gamma = np.sqrt(n + lamb)
    w = (1.0/(2*(n + lamb)) * np.ones(2*n)).reshape((-1,1)) # sigma point weights for remaining n*2 terms
    wm0 = lamb/(n + lamb)                                   # sigma point weight 0 for mean
    wc0 = lamb/(n + lamb) + (1 - alpha**2 + beta)           # sigma point weight 0 for covariance

    for t in tqdm(range(T_dur)):
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

        # Store.
        xh_t[:,t:t+1] = xh
        Ph_t[t,:,:] = RP @ RP.T
        iou_t.append(util.iou(z_t[:,t], zh))

        # propagate
        sigma_points = spt(xh, RP*gamma)            # generate n*2 sigma points again

        X0 = phi(xh)                                # propagate mean through dynamics
        X = phi(sigma_points)                       # propagate sigma points through dynamics

        wm = np.vstack((np.array([[wm0]]), w))      # stacked mean weights
        sig = np.hstack((X0, X))                    # stacked sigma points with mean sigma point
        xh = sig @ wm                               # predicted state mean, weighted sum of propagated sigma points

        diff_prop = np.sqrt(w[:,0])*(X-xh)          # covariance weighted sum of sigma points minus mean
        QR_prop = np.hstack((diff_prop, RQd))       # build the matrix for qr decomposition, adding the cholesky Q matrix side by side to the spread matrix
        S = np.linalg.qr(QR_prop.T, mode="r")       # perform QR decomposition again. combine dynamics and process noise.
        S = util.cholup(S.T, (X0-xh)[:,0], wc0)     # now we have the full square-root predicted state covariance, telling us how uncertain our predicted state is
                                                    # again we have to do the cholup because wc0 may be negative. We do S.T again because it expects a lower triangular
        RP = S

    return xh_t, Ph_t, iou_t

xh_t, Ph_t, iou_t = estimate(z_t, T, z_t.shape[1])

iou_t = np.array(iou_t)
print(np.mean(iou_t))
time = np.arange(0, z_t.shape[1]*T, T)
# util.plot_results(time, xh_t, Ph_t, 'my_ukf')

g_camera = np.array([0, 8.48704896, 4.9]) 
ani = animation.animate_ball_and_hook(xh_t, g_camera) 