import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import math

# Constants
N = 50 # number of tiles
C = 5 # number of possible colors


def markov_transition(pm):
    # Built Markov transition matrix.
    M_col = np.zeros(N)
    M_col[[-1, 0, 1]] = pm # last, first, and second
    M = sp.linalg.circulant(M_col)
    return M


def gen_truth(K, pc, pm):
    # Build the PMF for color correctness.
    pz = np.ones((C, C)) * (1 - pc)/(C - 1)
    for c in range(C):
        pz[c, c] = pc

    # Built Markov transition matrix.
    M = markov_transition(pm)

    # Make the map of tile colors.
    map = np.random.randint(0, C, N)
    get_pz_x(pc, map)

    # Pick an initial position.
    x = np.random.randint(0, N)

    # Allocate memory for storage.
    x_t = np.zeros(K)
    z_t = np.zeros(K, dtype=int)

    # For every time sample,
    for k in range(K):
        # Get a measurement based on the value of x.
        z = map[x] # the true color
        zh = np.random.choice(C, p=pz[z]) # the measurement

        # Store values.
        x_t[k] = x
        z_t[k] = zh

        # Get p(x) using the Markov transition matrix.
        px = M[:, x]

        # Choose a new x based on p(x).
        x = np.random.choice(N, p=px)

    return x_t, z_t, map


def get_pz_x(pc, map):
    # Get p(z|x) for every possible value of z.
    pz_x = np.ones((C, len(map))) * (1 - pc)/(C - 1)
    for c in range(C):
        n = np.where(map == c)[0]
        pz_x[c, n] = pc
        pz_x[c, :] /= pz_x[c, :].sum()
    return pz_x


def plot_tiles(map):
    R = 10.0
    N = len(map)
    dphi = 2*np.pi/N

    R_i = R - R*dphi/2 + R*dphi/20
    R_o = R + R*dphi/2 - R*dphi/20
    phi_a = np.arange(N)*dphi + dphi/20
    phi_b = phi_a + 18*dphi/20
    ca = np.cos(phi_a)
    sa = np.sin(phi_a)
    cb = np.cos(phi_b)
    sb = np.sin(phi_b)

    colors = ['tab:blue', 'tab:orange',
            'tab:green', 'tab:red', 'tab:purple']
    plt.plot([R_i - R*dphi/4, R_i - 1.5*R*dphi], [0, 0], color='black')
    plt.text(R_i - 1.5*R*dphi, 0, '0', ha='right', va='bottom')
    plt.text(R_i - 1.5*R*dphi, 0, str(N-1), ha='right', va='top')
    for n in range(N):
        color = colors[map[n]]
        plt.fill([R_i*ca[n], R_i*cb[n], R_o*cb[n], R_o*ca[n]],
                [R_i*sa[n], R_i*sb[n], R_o*sb[n], R_o*sa[n]],
                color=color, edgecolor=None)
    plt.axis('equal')
    plt.axis('off')


def plot_results(px_t, xh_t, x_t=None, color='tab:blue'):
    # Plot p(x) over time along with the truth if provided.
    plt.figure(figsize=(3.4, 2.101))
    plt.subplots_adjust(bottom=0.2)
    plt.rcParams.update({'text.usetex': True,
            'font.family': 'serif',
            'font.serif': 'cm',
            'font.size': '9'})
    N, K = px_t.shape
    for k in range(K):
        for n in range(N):
            if px_t[n, k] < 0.01:
                continue
            plt.fill([k-0.5, k+0.5, k+0.5, k-0.5],
                    [n-0.5, n-0.5, n+0.5, n+0.5],
                    color=color, edgecolor=None,
                    alpha=math.sqrt(px_t[n, k]))
    plt.plot(xh_t, color='tab:orange', label='$\\hat{x}$')
    if x_t is not None:
        plt.plot(x_t, color='tab:green', label='$x$')
    plt.legend()
    plt.xlabel('Time sample')
    plt.ylabel('Tile index')
    plt.xlim((0, K))