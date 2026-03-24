# Created by Daniel Jacobs, edited by David Woodburn.
import matplotlib.pyplot as plt
import numpy as np


def iou(z, zh):
    """
    Find the intersection over union using the radii and centers.

    Parameters
    ----------
    z : (3,) array like
        Array of radius and (u,v) center.
    zh : (3,) array like
        Array of radius and (u,v) center.

    Returns
    -------
    iou : float
        Intersection over union of areas.

    Notes
    -----
    https://mathworld.wolfram.com/Circle-CircleIntersection.html
    """
    # Parse the vectors.
    r, u, v = z
    rh, uh, vh = zh
    
    # Distance between centers
    d = np.sqrt((uh - u)**2 + (vh - v)**2)

    # No intersection
    if r + rh <= d:
        return 0

    # Get the areas.
    A = np.pi*r**2
    Ah = np.pi*rh**2
    total_area = A + Ah

    # One circle within the other
    if r >= d + rh:
        return Ah/A
    if rh >= d + r:
        return A/Ah
    
    # Distances from centers to bisecting chord
    l = (r**2 - rh**2 + d**2)/(2*d)
    lh = d - l

    # Areas of the circular segments
    S = r**2*np.arccos(l/r) - l*np.sqrt(r**2 - l**2)
    Sh = rh**2*np.arccos(lh/rh) - lh*np.sqrt(rh**2 - lh**2)

    # Get the intersectional area.
    intersection = S + Sh
    union = total_area - intersection

    return intersection/union

def cholup(R, v, w):
    n = np.size(v)
    for i in range(n):
        r = np.sqrt(R[i, i]**2 + w*v[i]**2)
        c = r/R[i, i]
        s = v[i]/R[i, i]
        R[i, i] = r
        R[i+1:n, i] = (R[i+1:n, i] + w*s*v[i+1:n])/c
        v[i+1:n]= c*v[i+1:n] - s*R[i+1:n, i]
    return R

def plot_results(t, xh_t, Ph_t, label_suffix=""):
    # Extract the elements of the state covariance matrix.
    s_px = np.sqrt(Ph_t[:, 0, 0])
    s_py = np.sqrt(Ph_t[:, 1, 1])
    s_pz = np.sqrt(Ph_t[:, 2, 2])
    s_hx = np.sqrt(Ph_t[:, 6, 6])
    s_hy = np.sqrt(Ph_t[:, 7, 7])
    s_hz = np.sqrt(Ph_t[:, 8, 8])

    # Ball position
    plt.figure()
    plt.fill_between(t, xh_t[0] - 3*s_px, xh_t[0] + 3*s_px, alpha=0.2)
    plt.fill_between(t, xh_t[1] - 3*s_py, xh_t[1] + 3*s_py, alpha=0.2)
    plt.fill_between(t, xh_t[2] - 3*s_pz, xh_t[2] + 3*s_pz, alpha=0.2)
    plt.plot(t, xh_t[0], label='$p_x$')
    plt.plot(t, xh_t[1], label='$p_y$')
    plt.plot(t, xh_t[2], label='$p_z$')
    plt.ylabel('Position (m)')
    plt.xlabel('Time, $t$ (s)')
    plt.legend()
    plt.title(f'Ball Position {label_suffix}')
    # plt.savefig(f'fig_ball_position_{label_suffix}.pdf')

    # Hook position
    plt.figure()
    plt.fill_between(t, xh_t[6] - 3*s_hx, xh_t[6] + 3*s_hx, alpha=0.2)
    plt.fill_between(t, xh_t[7] - 3*s_hy, xh_t[7] + 3*s_hy, alpha=0.2)
    plt.fill_between(t, xh_t[8] - 3*s_hz, xh_t[8] + 3*s_hz, alpha=0.2)
    plt.plot(t, xh_t[6], label='$h_x$')
    plt.plot(t, xh_t[7], label='$h_y$')
    plt.plot(t, xh_t[8], label='$h_z$')
    plt.ylabel('Position (m)')
    plt.xlabel('Time, $t$ (s)')
    plt.legend()
    plt.title(f'Hook Position {label_suffix}')
    # plt.savefig(f'fig_hook_position_{label_suffix}.pdf')

    # Ball velocity
    plt.figure()
    plt.fill_between(t, xh_t[3] - 3*s_px, xh_t[3] + 3*s_px, alpha=0.2)
    plt.fill_between(t, xh_t[4] - 3*s_py, xh_t[4] + 3*s_py, alpha=0.2)
    plt.fill_between(t, xh_t[5] - 3*s_pz, xh_t[5] + 3*s_pz, alpha=0.2)
    plt.plot(t, xh_t[3], label='$p_x$')
    plt.plot(t, xh_t[4], label='$p_y$')
    plt.plot(t, xh_t[5], label='$p_z$')
    plt.ylabel('Velocity (m/s)')
    plt.xlabel('Time, $t$ (s)')
    plt.legend()
    plt.title(f'Ball Velocity {label_suffix}')
    # plt.savefig(f'fig_ball_velocity_{label_suffix}.pdf')

    # 3D path
    plt.figure()
    ax = plt.axes(projection='3d')
    plt.plot(xh_t[0], xh_t[2], -xh_t[1], label='ball')
    plt.plot(xh_t[6, -1], xh_t[8, -1], -xh_t[7, -1], 'o', label='hook')
    ax.set_xlabel('$x$-axis (m)')
    ax.set_ylabel('$z$-axis (m)')
    ax.set_zlabel('$y$-axis (m)')
    plt.axis('equal')
    plt.legend()
    plt.title(f'Path {label_suffix}')
    # plt.savefig(f'fig_path_{label_suffix}.pdf')
    # plt.show()
