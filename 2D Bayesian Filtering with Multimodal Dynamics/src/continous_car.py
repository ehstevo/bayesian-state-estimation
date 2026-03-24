import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d as conv
from tqdm import tqdm
import util

# constants
K = 10                      # time steps
v = 0.6                     # initial scalar state variance
q = 0.1                     # scalar process noise variance
r = 4                       # measurement noise variance
p_lsr = [0.3, 0.4, 0.3]     # array of probabilities of moving left, straight, or right
lsr = np.array([[-3, 4], 
       [0, 5], 
       [3, 4]]  )           # matrix of motion coordinates. each row corresponds to left, straight, or right movement
dx = 0.2

# get truth and measurements
x_t, z_t = util.gen_data(K, v, q, r, p_lsr, lsr)

def estimate(K, x_t, z_t, dx):
    x_limits = util.domain(x_t[0], x_t[1], q, dx)       # from generated truth, minimum and maximum values of x and y +/- 4*std(process noise)
    z_limits = util.domain(z_t[0], z_t[1], r, dx)       # from measurements, minimum and maximum values of x and y +/- 4*std(measurments)
    limits = [min(x_limits[0], z_limits[0]), max(x_limits[1], z_limits[1]),
              min(x_limits[2], z_limits[2]), max(x_limits[3], z_limits[3])]         # union of x and z limits to create full mesh simulation environment
    X, Y = util.mesh(limits, dx)                        # mesh grid from x and y limits that encompasses the entire simulation
    px = util.norm2d(X, Y, [0,0], v)                    # intial state pdf
    px /= np.sum(px)*dx                                 # normalize

    vxy_limits = [-5, 5, -7, 7]                         # x and y limits for the process noise
    vX, vY = util.mesh(vxy_limits, dx)                  # mesh grid from vx and vy limits
    vx_left = util.norm2d(vX, vY, lsr[0], q)            # pdf for movement left
    vx_straight = util.norm2d(vX, vY, lsr[1], q)        # pdf for movement straight
    vx_right = util.norm2d(vX, vY, lsr[2], q)           # pdf for movement right
    process_noise_pdf = p_lsr[0]*vx_left + p_lsr[1]*vx_straight + p_lsr[2]*vx_right # full process noise pdf
    process_noise_pdf /= np.sum(process_noise_pdf)*dx

    # storage
    px_prior_t = np.zeros((K, X.shape[0], X.shape[1]))
    px_posterior_t = np.zeros((K, X.shape[0], X.shape[1]))

    for k in tqdm(range(K)):
        # store prior
        px_prior_t[k] = px

        # update
        pz_x = util.norm2d(X, Y, z_t[:,k], r)           # likelihood function
        posterior = pz_x * px                           # posterior = likelihood * prior
        posterior /= np.sum(posterior)*dx*dx            # normalize

        # store posterior
        px_posterior_t[k] = posterior

        # propagate
        px = conv(posterior, process_noise_pdf, mode='same')   # prior = conv(conv(dynamics, process noise), posterior))
                                                               # in this case, there are no deterministic dynamics, so
                                                               # prior = conv(process noise, posterior)
        px /= np.sum(px)*dx*dx                                 # normalize

    util.plot_results(X, Y, px_prior_t, px_posterior_t, z_t, x_t)
    plt.show()

estimate(K, x_t, z_t, dx)

