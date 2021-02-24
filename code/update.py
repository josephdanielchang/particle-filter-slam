'''
ECE276A WI21 PR2: Particle Filter SLAM
'''

import numpy as np
from pr2_utils import mapCorrelation

## Estimate the robot trajectory via differential drive motion model

def pose_update(MAP, particle_state, particle_weight, ranges, angles, transform_lidar2body, particle_count):

    # grid cells representing walls with 1
    map_wall = ((1 - 1 / (1 + np.exp(MAP['map']))) > 0.5).astype(np.int)

    x_im = np.arange(MAP['xmin'], MAP['xmax'] + MAP['res'], MAP['res'])  # x index of each pixel on log-odds map
    y_im = np.arange(MAP['ymin'], MAP['ymax'] + MAP['res'], MAP['res'])  # y index of each pixel on log-odds map

    # 9x9 grid around particle
    x_range = np.arange(-4 * MAP['res'], 5 * MAP['res'], MAP['res'])  # x deviation
    y_range = np.arange(-4 * MAP['res'], 5 * MAP['res'], MAP['res'])  # y deviation

    # end point of lidar ray in lidar frame (xy)
    ex = ranges * np.cos(angles)
    ey = ranges * np.sin(angles)

    # convert end point to 2D (xy)
    exy = np.ones((4, np.size(ex)))
    exy[0, :] = ex
    exy[1, :] = ey

    # transform end point to body frame
    exy = np.dot(transform_lidar2body, exy)

    correlation = np.zeros(particle_count)

    for i in range(particle_count):

        # current particle pose
        x_t = particle_state[:, i]
        x_w = x_t[0]
        y_w = x_t[1]
        theta_w = x_t[2]

        # transform end point to world frame
        transform_body2world = np.array([[np.cos(theta_w), -np.sin(theta_w), 0, x_w],
                                        [np.sin(theta_w), np.cos(theta_w), 0, y_w],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]])
        s_w = np.dot(transform_body2world, exy)
        ex_w = s_w[0, :]
        ey_w = s_w[1, :]
        Y = np.stack((ex_w, ey_w))

        # calculate correlation
        c = mapCorrelation(map_wall, x_im, y_im, Y, x_range, y_range)

        # find largest correlation
        correlation[i] = np.max(c)

    # update particle weight with softmax function
    d = np.max(correlation)
    beta = np.exp(correlation - d)
    p_h = beta / beta.sum()
    particle_weight *= p_h / np.sum(particle_weight * p_h)

    return particle_state, particle_weight



