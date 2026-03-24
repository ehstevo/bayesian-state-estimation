import numpy as np
import matplotlib.pyplot as plt


def gen_data(K, v, q, r, p_lsr, lsr):
    """
    Generate true positions, `x_t`, and noisy measurements, `z_t`.

    Parameters
    ----------
    K : int
        Number of time samples.
    v : float
        Variance of initial position.
    q : float
        Variance of process noise.
    r : float
        Variance of measurement noise.
    p_lsr : (3,) np.ndarray
        Array of probabilities of moving forward left, straight, or right.
    lsr : (3, 2) np.ndarray
        x and y coordinates of forward left, straight, and right motions.

    Returns
    -------
    x_t : (2, K) np.ndarray
        Time-history true coordinates.
    z_t : (2, K) np.ndarray
        Time-history noisy measurements.
    """

    # Define constants.
    P = np.eye(2)*v # initial state covariance
    Q = np.eye(2)*q # process noise covariance
    R = np.eye(2)*r # measurement noise covariance

    # Initialize the state.
    x = np.random.multivariate_normal(np.zeros(2), P)

    # Allocate storage.
    x_t = np.zeros((2, K))
    z_t = np.zeros((2, K))

    # For each point in time,
    for k in range(K):
        # Measurement
        z = np.random.multivariate_normal(x, R)

        # Storage
        x_t[:, k] = x
        z_t[:, k] = z

        # Propagate
        j = np.random.choice(len(p_lsr), p=p_lsr)
        nu = np.random.multivariate_normal(np.zeros(2), Q)
        x += lsr[j] + nu

    return x_t, z_t


def norm2d(X, Y, mu, q):
    """
    Return the 2D normal distribution over the domain (X, Y).

    Parameters
    ----------
    X : (J, I) np.ndarray
        Matrix of x-axis coordinates.
    Y : (J, I) np.ndarray
        Matrix of y-axis coordinates.
    mu : (2,) np.ndarray
        Vector of x and y mean values.
    q : float
        Variance in both x and y directions.

    Returns
    -------
    : (J, I) np.ndarray
        Matrix of probability density values.
    """
    return 1.0/(2*np.pi*q)*np.exp(-0.5 *
            ((X - mu[0])**2 + (Y - mu[1])**2)/q)


def domain(x, y, q, dxy):
    """
    Using the coordinates defined in `x` and `y`, add 3 standard deviations in
    all directions, quantize to the step size of coordinates, and return the
    limits.

    Parameters
    ----------
    x : (K,) np.ndarray
        Vector of x coordinates.
    y : (K,) np.ndarray
        Vector of y coordinates.
    q : float
        Variance in the x or y axis.
    dxy : float
        Step size in the x or y axis.

    Returns
    -------
    [x_min, x_max, y_min, y_max] : float list
        List of limits.
    """
    s = np.sqrt(q)
    x_min = round((np.min(x) - 4*s)/dxy)*dxy
    x_max = round((np.max(x) + 4*s)/dxy)*dxy
    y_min = round((np.min(y) - 4*s)/dxy)*dxy
    y_max = round((np.max(y) + 4*s)/dxy)*dxy
    return [x_min, x_max, y_min, y_max]


def mesh(limits, dx):
    """
    Create a mesh grid of x and y coordinates.

    Parameters
    ----------
    limits : (4,) np.ndarray
        x and y-axis limits: [x_min, x_max, y_min, y_max].
    dx : float
        Step size in the x or y axis.

    Returns
    -------
    X : (J, I) np.ndarray
        Matrix of x-axis coordinates where values increase from column to
        column.
    Y : (J, I) np.ndarray
        Matrix of y-axis coordinates where values increase from row to row.
    """
    Nx = round((limits[1] - limits[0])/dx) + 1
    Ny = round((limits[3] - limits[2])/dx) + 1
    x = np.linspace(limits[0], limits[1], Nx)
    y = np.linspace(limits[2], limits[3], Ny)
    X, Y = np.meshgrid(x, y)
    return X, Y


def plot_results(X, Y, prior_t, post_t=None, z_t=None, x_t=None):
    """
    Plot the posterior and prior PDFs and the measurements.

    Parameters
    ----------
    X : (J, I) np.ndarray
        Matrix of x-axis coordinates where values increase from column to
        column.
    Y : (J, I) np.ndarray
        Matrix of y-axis coordinates where values increase from row to row.
    prior_t : (K, J, I) np.ndarray
        Stack of K matrices, each J by I. Each matrix (layer of the stack) is
        the prior PDF for the corresponding moment in time.
    post_t : (K, J, I) np.ndarray, default None
        Stack of K matrices, each J by I. Each matrix (layer of the stack) is
        the posterior PDF for the corresponding moment in time.
    z_t : (2, K) np.ndarray, default None
        Time-history noisy measurements.
    """

    # Get the number of time samples.
    K = prior_t.shape[0]

    # For each moment in time,
    xmin = X.min()
    xmax = X.max()
    ymin = Y.min()
    ymax = Y.max()
    plt.figure(figsize=(5, 0.85*5*(ymax - ymin)/(xmax - xmin)))
    for k in range(K):
        # Plot the posterior.
        if post_t is not None:
            post = post_t[k]
            post /= post.max()
            post = post*(post > 0.1)
            plt.contour(X, Y, post, 3, cmap=plt.cm.Greens)

        # Plot the posterior.
        prior = prior_t[k]
        prior /= prior.max()
        prior = prior*(prior > 0.1)
        plt.contour(X, Y, prior, 5, cmap=plt.cm.Oranges)

    # Plot the measurements.
    if z_t is not None:
        plt.plot(z_t[0, :], z_t[1, :], 'o', color='tab:blue')

    if x_t is not None:
        plt.plot(x_t[0, :], x_t[1, :], 'o', color='tab:purple')

    # Add axis labels and grid.
    plt.grid(linewidth=0.1)
    plt.xlabel("$x$-axis (m)")
    plt.ylabel("$y$-axis (m)")

    # Set layout.
    plt.axis("equal")
    plt.tight_layout()

    # Add legend.
    if post_t is not None or z_t is not None:
        ax = plt.gca()
        xlims = ax.get_xlim()
        ylims = ax.get_ylim()
        Dx = xlims[1] - xlims[0]
        Dy = ylims[1] - ylims[0]
        dx = 1.5
        dy = 0.5
        x = xlims[0] + 2*dx
        y = ylims[0] + dy
        plt.fill([x, x + dx, x + dx, x], [y, y, y + dy, y + dy],
                color='tab:orange', edgecolor=None)
        plt.text(x + 2, y, '$x^-$')
        if prior_t is not None:
            x += Dx/3
            plt.fill([x, x + dx, x + dx, x], [y, y, y + dy, y + dy],
                    color='tab:green', edgecolor=None)
            plt.text(x + 2, y, '$x^+$')
        if z_t is not None:
            x += Dx/3
            plt.fill([x, x + dx, x + dx, x], [y, y, y + dy, y + dy],
                    color='tab:blue', edgecolor=None)
            plt.text(x + 2, y, '$z$')
        if x_t is not None:
            x += Dx/3
            plt.fill([x, x + dx, x + dx, x], [y, y, y + dy, y + dy],
                    color='tab:purple', edgecolor=None)
            plt.text(x + 2, y, '$x$')
