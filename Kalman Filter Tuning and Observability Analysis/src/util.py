import scipy.linalg as la
import numpy as np
import matplotlib.pyplot as plt

# constants
T = 0.1                                 # sampling period (s)
T_dur = 100                             # simulation duration (s)
theta = np.pi/4
# gamma = (4*np.pi)/180

# system definition
F = np.zeros((4, 4))                    # State transition matrix (dynamics)
# F[0, 1] = gamma
F[0, 2] = 1
# F[1, 0] -1*gamma
F[1, 3] = 1
# F[2, 3] = gamma
# F[3, 2] = -1*gamma

B = np.zeros((4, 2))                    # Control input matrix
B[2, 0] = 1
B[3, 1] = 1

H = np.zeros((2, 4))                    # Observation matrix
H[0, 0] = np.cos(theta)
H[0, 1] = np.sin(theta)
H[1, 0] = np.sin(theta)*-1
H[1, 1] = np.cos(theta)
# H[1,2] = 1

Q = np.diag([0.01, 0.01, 0.01, 0.01])   # Process noise covariance matrix
R = np.diag([1.0, 0.1])                 # Measurement noise covariance matrix

# vanloan method to discretize F, B, and Q into Phi, Bd, and Qd
def vanloan(F, B, Q, T):
    # Phi
    N = F.shape[1]
    Phi = la.expm(F*T)

    # Bd
    M = B.shape[1]
    G = np.vstack((np.hstack((F, B)), np.zeros((M, N+M))))
    H = la.expm(G*T)
    Bd = H[0:N, N:(N+M)]

    # Qd
    L = np.vstack((np.hstack((-F, Q)), np.hstack((np.zeros((N, N)), F.T))))
    H = la.expm(L*T)
    Qd = Phi @ H[0:N, N:(2*N)]

    return Phi, Bd, Qd

def gen_truth(Phi, Bd, Qd, H, R, T, T_dur):
    # Storage
    K = int(T_dur/T)
    x_t = np.zeros((4, K))
    u_t = np.zeros((2, K))
    z_t = np.zeros((2, K))

    x = np.array([0, 0, 0.8, 0]).T.reshape(4, 1)                  # initial state

    for k in range(K):
        x_t[:,k:k+1] = x                                          # store x value at time step k
        a_x = -1*np.sin(((2*np.pi)/5)*(k*T))                      # acceleration input in x direction
        a_y = -1*np.cos(((2*np.pi)/10)*(k*T))                     # acceleration input in y direction
        u_t[:,k:k+1] = np.array([a_x, a_y]).reshape(2, 1)         # store input values at time step k
        x = (Phi @ x) + (Bd @ u_t[:,k:k+1]) \
             + np.random.multivariate_normal(np.zeros(4), Qd).reshape(4, 1)             # propagate x based on dynamics, control inputs, and nosie
        
        z_t[:,k:k+1] = (H @ x_t[:,k:k+1]) \
                        + np.random.multivariate_normal(np.zeros(2), R).reshape(2, 1)   # calculate measurement based on x before propagation

    data = np.vstack((x_t, u_t, z_t))
    data.T.tofile("data/xuz.bin")

    return x_t, u_t, z_t


def plot_paths(truth, estimated, measurements=None):
    # extract position data
    true_px = truth[0,:]
    true_py = truth[1,:]

    est_px = estimated[0,:]
    est_py = estimated[1,:]

    # convert measurement data from sensor frame to coordinate frame
    measurements = H.T @ measurements

    # create plot
    plt.figure(figsize=(12, 9))

    # plot truth
    plt.plot(true_px, true_py, 'g-', label='true path', linewidth=2)

    # plot estimate plath
    plt.plot(est_px, est_py, 'b--', label='estimated path', linewidth=2)

    # mark start and end points
    plt.scatter(true_px[0], true_py[0], color='green', s=150, zorder=5, marker='o')
    plt.scatter(true_px[-1], true_py[-1], color='red', s=150, zorder=5, marker='x')

    if measurements is not None:
        plt.scatter(measurements[0,:], measurements[1,:], color='orange', s=10, marker='.', label='measurements')    

    plt.title('True vs. Estimated Trajectory')
    plt.xlabel('x-axis position (m)')
    plt.ylabel('y-axis position (m)')
    plt.legend()
    plt.grid(True)

    plt.show()

def plot_errors(xe_t, P_t):
    # time vector
    t = np.arange(0, 100, 0.1)

    # calculate "actual" standard deviation
    truth_std = np.std(xe_t, axis=1)

    # calculate expected standard deviation
    expected_std = np.sqrt(P_t)

    # plot
    state_labels = ['$p_x$ Error (m)', '$p_y$ Error (m)', '$v_x$ Error (m/s)', '$v_y$ Error (m/s)']
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))

    for i in range(4):
        ax = axes[i]

        # plot error
        ax.plot(t, xe_t[i,:], 'b-', label='error')

        # plot expected 3-sigma bounds
        ax.fill_between(t, 3*expected_std[i,:], -3*expected_std[i,:], alpha=0.4)

        # plot actual 3-sigma bounds
        ax.axhline(y= 3 * truth_std[i], color='r', label='true 3-sigma')
        ax.axhline(y = -3 * truth_std[i], color='r')

        ax.set_ylabel(state_labels[i])
        ax.set_xlabel('Time, $t$ (s)')
        ax.grid(True)
        ax.legend(loc='upper right')

        ax.set_xlim(left = 0, right = 100)
    
    fig.suptitle('Error Analysis')
    plt.show()

def plot_results(E):
    e = np.flip(E, 0)
    e0 = e[3, 3]
    de = e.max() - e0
    plt.figure()
    plt.pcolormesh(e, cmap='bwr', vmin=e0-de, vmax=e0+de)
    for nq in range(7):
        for nr in range(7):
            plt.text(nr + 0.5, nq + 0.5, round(e[nq, nr], 2),
                    horizontalalignment='center',
                    verticalalignment='center_baseline',
                    fontsize=18, fontfamily='serif')
            plt.axis('off')
            plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0)
    plt.show()
    

# Phi, Bd, Qd = vanloan(F, B, Q, T)
# x_t, u_t, z_t = gen_truth(Phi, Bd, Qd, H, R, T, T_dur)