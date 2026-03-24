import numpy as np
import scipy.linalg as la
import util
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


# Jacobian matrix Phi
def calculate_jacobian_Phi(state, ag):
    # variables
    ag = ag.reshape((3, 1))
    s_x = state[0][0]-state[6][0]
    s_y = state[1][0]-state[7][0]
    s_z = state[2][0]-state[8][0]
    s_ = np.array([[s_x], [s_y], [s_z]])
    s = np.linalg.norm(s_)
    vx = state[3][0]
    vy = state[4][0]
    vz = state[5][0]
    v_ = np.array([[vx], [vy], [vz]])
    v = np.linalg.norm(v_)

    # d ag
    ag_px = -2*(s_x/(s**2))*ag - (1/(s**2))*np.array([[2*g[0]*s_x + g[1]*s_y + g[2]*s_z], [g[0]*s_y], [g[0]*s_z]])
    ag_py = -2*(s_y/(s**2))*ag - (1/(s**2))*np.array([[g[1]*s_x], [g[0]*s_x + 2*g[1]*s_y + g[2]*s_z], [g[1]*s_z]])
    ag_pz = -2*(s_z/(s**2))*ag - (1/(s**2))*np.array([[g[2]*s_x], [g[2]*s_y], [g[0]*s_x + g[1]*s_y + 2*g[2]*s_z]])
    ag_hx = -1 * ag_px
    ag_hy = -1 * ag_py
    ag_hz = -1 * ag_pz

    # d ac
    ac_px = -(v**2)/(s**2) * (np.array([[1],[0],[0]])-(2*(s_x/(s**2))*s_))
    ac_py = -(v**2)/(s**2) * (np.array([[0],[1],[0]])-(2*(s_y/(s**2))*s_))
    ac_pz = -(v**2)/(s**2) * (np.array([[0],[0],[1]])-(2*(s_z/(s**2))*s_))
    ac_hx = -1 * ac_px
    ac_hy = -1 * ac_py
    ac_hz = -1 * ac_pz
    ac_vx = -2*(s_/(s**2))*vx
    ac_vy = -2*(s_/(s**2))*vy
    ac_vz = -2*(s_/(s**2))*vz

    # d ad
    ad_v = (k_d/v)*((v**2)*np.eye(3) + (v_ @ v_.T))
    ad_vx = ad_v[:,0].reshape((3, 1))
    ad_vy = ad_v[:,1].reshape((3, 1))
    ad_vz = ad_v[:,2].reshape((3, 1))

    # da / dx
    a_px = ag_px + ac_px
    a_py = ag_py + ac_py
    a_pz = ag_pz + ac_pz
    a_vx = ac_vx + ad_vx
    a_vy = ac_vy + ad_vy
    a_vz = ac_vz + ad_vz
    a_hx = ag_hx + ac_hx
    a_hy = ag_hy + ac_hy
    a_hz = ag_hz + ac_hz
    da_dx = np.hstack((a_px, a_py, a_pz, a_vx, a_vy, a_vz, a_hx, a_hy, a_hz))

    # phi = I_9 + T*[[0_3 I_3 0_3], [0_3 0_3 0_3], [0_3 0_3 0_3]] + [[1/2T^2*da/dx], [T*da/dx], [0_3x9]]
    first = np.eye(9)
    second = np.zeros((9,9))
    second[0][3] = 1
    second[1][4] = 1
    second[2][5] = 1
    second = second * T
    third = np.vstack((0.5*(T**2)*da_dx, T*da_dx, np.zeros((3, 9))))
    Phi = first + second + third

    return Phi

# Jacobian matrix H
def calculate_jacobian_H(state):
    px = state[0][0]
    py = state[1][0]
    pz = state[2][0]
    H = np.zeros((3, 9))
    H[0][2] = -k*(r_b/(pz**2))
    H[1][0] = k/pz
    H[1][2] = -k*(px/(pz**2))
    H[2][1] = k/pz
    H[2][2] = -k*(py/(pz**2))
    return H

# measurements
z_t = np.fromfile("data/z_t1.bin").reshape((-1, 3)).T
z_t2 = np.fromfile("data/z_t2.bin").reshape((-1, 3)).T
z_t3 = np.fromfile("data/z_t3.bin").reshape((-1, 3)).T

# define process and measurement noise covariance matrices
Qd = np.eye(9)*0.0001*T
R = np.eye(3)*0.001

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
    
    # storage
    xh_t = np.zeros((9, T_dur))                                 # estimates
    Ph_t = np.zeros((T_dur, 9, 9))                              # covariances
    iou_t = []
    
    for t in tqdm(range(T_dur)):
        # update
        H = calculate_jacobian_H(xh)                            # linearize non-linear measurement model
                                                                # relates a small change in the state vector to the corresponding change 
                                                                # in the predicted measurement
        S = H @ P @ H.T + R
        K = P @ H.T @ la.inv(S)                                 # Kalman gain - weighting factor that balances predicted states's uncertainty
                                                                # against the measurement's uncertainty
        r_p = k*(r_b/xh[2][0])                                  # calculate predicted r_p measurement from state
        u = c_x + k*(xh[0][0]/xh[2][0])                         # calculate predicted u measurement from state
        v = c_y + k*(xh[1][0]/xh[2][0])                         # calculate predicted v measurement from state
        zh = np.array([[r_p], [u], [v]])                        # predicted measurement
        r = z_t[:,t:t+1] - zh                                   # residual
        xh = xh + (K @ r)                                         # new estimate = old estimate + gain * residual
        P = P - (K @ H @ P)                                     # update state covariance matrix

        # store
        xh_t[:,t:t+1] = xh
        Ph_t[t,:,:] = P
        iou_t.append(util.iou(z_t[:,t], zh[:,0]))

        # propagate
        # calculate accelerations
        ag, ac, ad = calculate_accelerations(xh)
        Phi = calculate_jacobian_Phi(xh, ag)                    # linearize non-linear system dynamics
                                                                # relates a small change in the current state to the corresponding change in 
                                                                # the predicted next state under the dynamics model
        a = ag + ac + ad + g
        px = xh[0][0]
        py = xh[1][0]
        pz = xh[2][0]
        vx = xh[3][0]
        vy = xh[4][0]
        vz = xh[5][0]
        xh[0][0] = px + vx*T + 0.5*(T**2)*a[0]                  # propagate px
        xh[1][0] = py + vy*T + 0.5*(T**2)*a[1]                  # propagate py
        xh[2][0] = pz + vz*T + 0.5*(T**2)*a[2]                  # propagate pz
        xh[3][0] = vx + T*a[0]                                  # propagate vx
        xh[4][0] = vy + T*a[1]                                  # propagate vy
        xh[5][0] = vz + T*a[2]                                  # propagate vz
        P = Phi @ P @ Phi.T + Qd                                # propagate state covariance matrix

    return xh_t, Ph_t, iou_t

xh_t, Ph_t, iou_t = estimate(z_t, T, z_t.shape[1])

iou_t = np.array(iou_t)
print(np.mean(iou_t))
time = np.arange(0, z_t.shape[1]*T, T)
# util.plot_results(time, xh_t, Ph_t, 'ekf')

g_camera = np.array([0, 8.48704896, 4.9]) 
ani = animation.animate_ball_and_hook(xh_t, g_camera) 