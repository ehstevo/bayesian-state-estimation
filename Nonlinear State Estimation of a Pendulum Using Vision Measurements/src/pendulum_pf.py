import numpy as np
import matplotlib.pyplot as plt
import util
from tqdm import tqdm
import animation

# constants
T = 0.01                                    # sampling period
J = 500                                     # number of particles
g = np.array([0, 8.48704896, 4.9]).T
k_d = 0.1                                   # drag coefficient
r_b = 12*(0.0254) / (2*np.pi)               # true radius of the ball
k = 900.0                                   # camera scaling factor
c_x = 640                                   # horizontal center of the camera image in pixels
c_y = 360                                   # vertical center of the camera image in pixels

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
    snorm = np.linalg.norm(s)
    snorm = max(snorm, 1e-3)
    u_s = s/snorm
    ag = -u_s*(np.dot(u_s, g))

    # calculate ac (centripetal acceleration)
    v = np.array([state[3][0], state[4][0], state[5][0]])
    ac = -u_s*(np.linalg.norm(v)**2/snorm)

    # calculate ad (drag acceleration)
    ad = -v*(k_d*np.linalg.norm(v))

    return ag, ac, ad

def phi(particles):
    phi = np.zeros_like(particles)
    second = np.zeros((9,9))
    second[0][3] = 1
    second[1][4] = 1
    second[2][5] = 1
    second = second * T
    third = np.zeros((9,1))
    for i in range(particles.shape[1]):
        x = particles[:,i:i+1]
        ag, ac, ad = calculate_accelerations(x)
        a = ag + ac + ad + g
        third = np.zeros((9,1))
        third = np.vstack((0.5*(T**2)*a[0], 0.5*(T**2)*a[1], 0.5*(T**2)*a[2], T*a[0], T*a[1], T*a[2], 0, 0, 0))
        phi[:,i:i+1] = x + (second @ x) + third
    return phi

def h(particles):
    x = particles[0,:].copy()
    y = particles[1,:].copy()
    z = particles[2,:].copy()
    # z = np.maximum(z, 0.8)
    rp_vec = k*(r_b / z)
    u_vec = c_x + k*(x/z)
    v_vec = c_y + k*(y/z)
    return np.vstack((rp_vec, u_vec, v_vec))

# process and measurement noise
Qd = np.eye(9)*0.001*T                     # Process noise covariance
R = np.eye(3)*10                         # Measurement noise covariance
R_inv = np.linalg.inv(R)
RQd = np.linalg.cholesky(Qd)

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
    P = np.eye(9)*0.001
    RP = np.linalg.cholesky(P)
    x_j = xh + RP @ np.random.randn(9, J)           # sample J particles from our initial pdf with mean = xh and std = RP
    # storage
    xh_t = np.zeros((9, T_dur))                     # estimates
    Ph_t = np.zeros((T_dur, 9, 9))                  # covariances
    iou_t = []
    Je_t = []

    logw = np.log(np.ones(J)/J)                     # initial log weights (np.exp() was unstable so we use log weights and likelihood)

    for t in tqdm(range(T_dur)):
        # update
        z = z_t[:,t:t+1]                            # get measurement
        zh = h(x_j)                                 # get predicted measurement for particles
        r = z - zh                                  # residuals for each particle
        maha = np.sum(r * (R_inv @ r), axis=0)      # one mahalanobis term per particle, essentially a square with uncertainty
                                                    # each term (i,j) in r is multiplied by the (i,j) term in (R_inv @ r)
                                                    # then we sum down each column (sum of squares for each residual)
        logw = logw - 0.5 * maha                    # weights * np.exp(-0.5*maha) = logweights - 0.5*maha when taking logs
        m = np.max(logw)                            # recenter for numerical stability
        w = np.exp(logw - m)                        # return logweights to weights
        w = np.maximum(w, 1e-5)                     # dont allow weight to go all the way to zero
        w = w / np.sum(w)                           # normalize weights
        logw = np.log(w)                            # return weights to logweights for update

        # estimate
        xh = x_j @ w.reshape((-1, 1))

        d = x_j - xh                               # sum of differences between estimate and particle state
        P = (w*d) @ d.T                             # estimate of covariance matrix for moment in time
        
        # store
        xh_t[:,t:t+1] = xh
        Ph_t[t,:,:] = P
        iou_t.append(util.iou(z_t[:,t], h(xh)[:,0]))

        # resample
        Je = 1.0 / np.sum(w**2)                     # effective number of samples
        Je_t.append(Je)

        if Je < (0.6 * J):                                   # decide if time to resample
            comb = (np.random.rand() + np.arange(J))/J
            je = np.searchsorted(np.cumsum(w), comb)
            x_j = x_j[:, je]                                 # does not create new states, duplicates highest weighted states and discards lowest weighted states (in theory)
            w = np.ones(J) / J                               # resets weight values
            logw = np.log(w)                                 # return weights to logweights for update

        # propagate
        nu_j = RQd @ np.random.randn(9, J)                  # sample random normal values for each particle with std cholesky(Q)
        x_j = phi(x_j) + nu_j                               # propagate particles and add noise

    return xh_t, Ph_t, iou_t, Je_t
    

xh_t, Ph_t, iou_t, Je_t = estimate(z_t, T, z_t.shape[1])

iou_t = np.array(iou_t)
Je_t = np.array(Je_t)
print(np.mean(iou_t))
# print(np.mean(Je_t))
time = np.arange(0, z_t.shape[1]*T, T)
# util.plot_results(time, xh_t, Ph_t, 'particle_filter')

g_camera = np.array([0, 8.48704896, 4.9]) 
ani = animation.animate_ball_and_hook(xh_t, g_camera) 
