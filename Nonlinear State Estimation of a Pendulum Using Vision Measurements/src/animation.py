# -*- coding: utf-8 -*- 
""" 
Created on Mon Mar 11 19:01:59 2024 

@author: Shawn Whitney 
""" 

import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation 
from scipy.spatial.transform import Rotation as R 

def animate_ball_and_hook(xh_t, g_camera): 

    def transform_coordinates(xh_t_sub, rotation): 
        # Apply the rotation to each point in the subset of xh_t 
        return rotation.apply(xh_t_sub.T).T # Transpose before and after for correct shape 

    # Normalize the gravity vector in camera frame 
    g_camera_normalized = g_camera / np.linalg.norm(g_camera) 
    # World frame gravity vector (assuming downward along Z-axis, normalized) 
    g_world_normalized = np.array([0, 0, -1]) 
    # Compute the rotation required to align the gravity vectors 
    rotation_axis = np.cross(g_camera_normalized, g_world_normalized) 
    rotation_angle = np.arccos(np.dot(g_camera_normalized, g_world_normalized)) 
    # Create the rotation object 
    rotation = R.from_rotvec(rotation_axis * rotation_angle) 

    # Transform the ball and hook positions to world frame 
    xh_t[:3, :] = transform_coordinates(xh_t[:3, :], rotation) 
    xh_t[6:9, :] = transform_coordinates(xh_t[6:9, :], rotation) 

    # Function to update the animation 
    def update(num, xh_t, ball_line, hook_point, string_line):
        max_history = 50 
        start_index = max(0, num - max_history) 
        ball_line.set_data(xh_t[0:2, start_index:num+1]) 
        ball_line.set_3d_properties(xh_t[2, start_index:num+1]) 
        hook_point.set_data([xh_t[6, num]], [xh_t[7, num]]) 
        hook_point.set_3d_properties(xh_t[8, num]) 
        string_line.set_data([xh_t[6, num], xh_t[0, num]], [xh_t[7, num], xh_t[1, num]]) 
        string_line.set_3d_properties([xh_t[8, num], xh_t[2, num]]) 
        return ball_line, hook_point, string_line 

    # Set up the figure for animation 
    fig = plt.figure() 
    ax = fig.add_subplot(111, projection='3d') 
    ball_line, = ax.plot(xh_t[0, 0:1], xh_t[1, 0:1], xh_t[2, 0:1]) 
    hook_point, = ax.plot([xh_t[6, 0]], [xh_t[7, 0]], [xh_t[8, 0]], 'o', markersize=10) 
    string_line, = ax.plot([xh_t[6, 0], xh_t[0, 0]], [xh_t[7, 0], xh_t[1, 0]], [xh_t[8, 0], xh_t[2, 0]], 'r-') 

    # Configure the axes 
    ax_limits = [np.min(xh_t[:9, :]), np.max(xh_t[:9, :])] 
    ax.set_xlim3d(ax_limits) 
    ax.set_ylim3d(ax_limits) 
    ax.set_zlim3d(ax_limits) 

    # Create and start the animation 
    ani = FuncAnimation(fig, update, frames=xh_t.shape[1], fargs=(xh_t, ball_line, hook_point, string_line), interval=10, blit=False, repeat=False) 

    plt.show() 
    return ani 

# Example usage: 
# g_camera = np.array([0, 8.48704896, 4.9]) 
# ani = animate_ball_and_hook(xh_t, g_camera) 
