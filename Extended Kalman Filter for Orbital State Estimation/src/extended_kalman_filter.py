import numpy as np
import util
import scipy.linalg as la

# constants
T = 1                               # sampling period (s)
T_dur = 28800                       # 8 hour duration of simulation in (s)
Ge = 3.986e14                       # gravitational acceleration on the satellite (m^3/s^2)
epsilon = 0.001                     # use to make sure we dont divide by 0

# system definition
# dynamics equations:
# f1 = px + vx*T
# f2 = py + vy*T
# f3 = vx + ax*T    where ax = -(Ge*px)/(px^2+py^2)^3/2
# f4 = vy + ay*T    where ay = -(Ge*py)/(px^2+py^2)^3/2

# Jacobian matrix Phi
def calculate_jacobian_Phi(current_state):
    px = current_state[0][0]
    py = current_state[1][0]
    Phi = np.zeros((4, 4))
    Phi[0,0] = 1                                                # df1/dpx
    Phi[0,1] = 0                                                # df1/dpy
    Phi[0,2] = T                                                # df1/dvx
    Phi[0,3] = 0                                                # df1/dvy
    Phi[1,0] = 0                                                # df2/dpx
    Phi[1,1] = 1                                                # df2/dpy
    Phi[1,2] = 0                                                # df2/dvx
    Phi[1,3] = T                                                # df2/dvy
    Phi[2,0] = (Ge*T*(2*(px**2)-(py**2)))/((px**2+py**2)**2.5)  # df3/dpx
    Phi[2,1] = (3*Ge*T*px*py)/((px**2+py**2)**2.5)              # df3/dpy
    Phi[2,2] = 1                                                # df3/dvx
    Phi[2,3] = 0                                                # df3/dvy
    Phi[3,0] = (3*Ge*T*px*py)/((px**2+py**2)**2.5)              # df4/dpx
    Phi[3,1] = (Ge*T*(2*(py**2)-(px**2)))/((px**2+py**2)**2.5)  # df4/dpy
    Phi[3,2] = 0                                                # df4/dvx
    Phi[3,3] = 1                                                # df4/dvy
    return Phi

# measurement equation:
# f = arctan2(px,py)

# Jacobian matrix H
def calculate_jacobian_H(current_state):
    px = current_state[0][0]
    py = current_state[1][0]
    H = np.zeros((1,4))
    H[0,0] = -py/(px**2+py**2)                                  # df/dpx
    H[0,1] = px/(px**2+py**2)                                   # df/dpy
    H[0,2] = 0                                                  # df/dvx
    H[0,3] = 0                                                  # df/dvy
    return H

# generate truth states and measurements
x_t, z_t = util.gen(T_dur, T)

# define Q and R
R = (0.2*np.pi/180)**2                                          # measurement noise variance
Qd = np.diag([0.1**2, 0.1**2, 0.05**2, 0.05**2])*T              # process noise covariance matrix

def estimate(x_t, z_t, T_dur, T):
    # initial state and covariance
    xh = np.array([[2e7], [0.0], [0], [4500]])
    P = np.diag([1000.0**2, 1000**2, 100**2, 100**2])

    # storage
    xh_t = np.zeros((4, T_dur))                                 # estimates
    Ph_t = np.zeros((T_dur, 4, 4))                              # covariances

    for k in range(T_dur):
        # update
        H = calculate_jacobian_H(xh)                            # linearize non-linear measurement model
                                                                # relates a small change in the state vector to the corresponding change 
                                                                # in the predicted measurement
        S = H @ P @ H.T + R
        K = P @ H.T @ la.inv(S)                                 # Kalman gain - weighting factor that balances predicted states's uncertainty
                                                                # against the measurement's uncertainty
        r = z_t[k] - np.arctan2(xh[1][0], xh[0][0])             # residual
        r = (r + np.pi) % (2*np.pi) - np.pi                     # corrects for wrap-around errors
        xh = xh + K*r                                           # new estimate = old estimate + gain * residual
        P = P - (K @ H @ P)                                     # update state covariance matrix

        # store
        xh_t[:,k:k+1] = xh
        Ph_t[k,:,:] = P

        # propagate
        Phi = calculate_jacobian_Phi(xh)                        # linearize non-linear system dynamics
                                                                # relates a small change in the current state to the corresponding change in 
                                                                # the predicted next state under the dynamics model
        px = xh[0][0]
        py = xh[1][0]
        vx = xh[2][0]
        vy = xh[3][0]
        xh[0][0] = px + vx*T                                    # propagate px
        xh[1][0] = py + vy*T                                    # propagate py
        xh[2][0] = vx + T*(-1*Ge*px)/((px**2+py**2)**1.5)       # propagate vx
        xh[3][0] = vy + T*(-1*Ge*py)/((px**2+py**2)**1.5)       # propagate vy
        P = Phi @ P @ Phi.T + Qd                                # propagate state covariance matrix

    return xh_t, Ph_t

xh_t, Ph_t = estimate(x_t, z_t, T_dur, T)


# Time range for plotting
t = np.arange(T_dur)
util.plot_results(t, x_t, xh_t, Ph_t)