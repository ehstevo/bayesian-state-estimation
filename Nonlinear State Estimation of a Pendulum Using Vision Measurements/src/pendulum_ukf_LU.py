import util
import numpy as np
import scipy.linalg as la
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

# measurements
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

def spt(sig, x, R):
    n = len(x)
    sig[:, :] = x
    sig[:,:n] += R
    sig[:,n:] -= R

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
sqrt_R = la.cholesky(R).T
sqrt_Qd = la.cholesky(Qd).T

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
    RP = la.cholesky(P)

    # storage
    sig = np.zeros((n, 2*n))
    xh_t = np.zeros((9, T_dur))                                 # estimates
    Ph_t = np.zeros((T_dur, 9, 9))                              # covariances
    iou_t = []

    # calculate scaling parameter lambda
    lambda_ = alpha**2 * (n+kappa) - n
    gamma = np.sqrt(n + lambda_)

    # calculate weights for mean and covariance
    w = 1.0/(2*(n + lambda_)) * np.ones(2*n)
    wm0 = lambda_/(n + lambda_)
    wc0 = lambda_/(n + lambda_) + (1 - alpha**2 + beta)

    # Initialize the augmented matrices.
    Lu = np.zeros((2*n + m, m + n))
    Lu[2*n:, :m] = sqrt_R
    Lp = np.zeros((3*n, n))
    Lp[2*n:] = sqrt_Qd

    for t in tqdm(range(T_dur)):
        # update
        z = z_t[:, t]

        # Sigma-point transform.
        spt(sig, xh, gamma*RP)

        # Update.
        Z0 = h(xh)
        Z = h(sig)
        zh = (wm0*Z0)[:,0] + np.sum(w*Z, axis=1)
        Lu[:2*n, :m] = np.sqrt(w[:, None])*(Z.T - zh)
        Lu[:n, m:] = RP.T/np.sqrt(2)
        Lu[n:2*n, m:] = -RP.T/np.sqrt(2)
        U = np.linalg.qr(Lu, mode="r")
        RS = U[:m, :m].T
        W = U[:m, m:].T
        RP = U[m:(m + n), m:].T
        Kg = W @ np.linalg.inv(RS)
        xh += Kg @ (z - zh).reshape((3,1))

        # Store.
        xh_t[:, t:t+1] = xh
        Ph_t[t, :, :] = RP @ RP.T
        iou_t.append(util.iou(z_t[:,t], zh))

        # Sigma-point transform.
        spt(sig, xh, gamma*RP)

        # Propagate.
        X0 = phi(xh)
        X = phi(sig)
        xh = ((wm0*X0)[:,0] + np.sum(w*X, axis=1)).reshape((9,1))
        Lp[:2*n] = np.sqrt(w[:, None])*(X - xh).T
        RP = np.linalg.qr(Lp, mode="r").T
        RP = util.cholup(RP, (X0 - xh)[:,0], wc0)

    return xh_t, Ph_t, iou_t

xh_t, Ph_t, iou_t = estimate(z_t, T, z_t.shape[1])

iou_t = np.array(iou_t)
print(np.mean(iou_t))
time = np.arange(0, z_t.shape[1]*T, T)
# util.plot_results(time, xh_t, Ph_t, 'ukf')

g_camera = np.array([0, 8.48704896, 4.9]) 
ani = animation.animate_ball_and_hook(xh_t, g_camera) 