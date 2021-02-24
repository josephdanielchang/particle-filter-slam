'''
ECE276A WI21 PR2: Particle Filter SLAM
'''

import numpy as np
from pr2_utils import bresenham2D
import matplotlib.pyplot as plt

# Update log-odds map according to lidar scan

def update_map(MAP, particle_max_state, ranges, angles, transform_lidar2body):

    # Body2World transform
    x_world = particle_max_state[0]
    y_world = particle_max_state[1]
    theta_world = particle_max_state[2]
    transform_body2world = np.array(
                          [[np.cos(theta_world), -np.sin(theta_world), 0, x_world],
                           [np.sin(theta_world),  np.cos(theta_world), 0, y_world],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])

    # start point of lidar ray in lidar frame (xy)
    sx = np.ceil((x_world - MAP['xmin']) / MAP['res']).astype(np.int16)-1
    sy = np.ceil((y_world - MAP['ymin']) / MAP['res']).astype(np.int16)-1

    # end point of lidar ray in lidar frame (xy)
    ex = ranges * np.cos(angles)
    ey = ranges * np.sin(angles)

    # convert end point to 3D (xyz)
    exyz = np.ones((4, np.size(ex)))
    exyz[0, :] = ex
    exyz[1, :] = ey
    exyz[2, :] = 0

    # transform end point to world frame
    exyz = np.dot(transform_lidar2body, exyz)
    exyz = np.dot(transform_body2world, exyz)

    # convert end point in world frame to cells
    ex = exyz[0, :]
    ey = exyz[1, :]
    ex = np.ceil((ex - MAP['xmin']) / MAP['res']).astype(np.int16)-1
    ey = np.ceil((ey - MAP['ymin']) / MAP['res']).astype(np.int16)-1

    # bresenham
    for i in range(np.size(ranges)):
        bresenham_points = bresenham2D(sx, sy, ex[i], ey[i])
        bresenham_points_x = bresenham_points[0, :].astype(np.int16)
        bresenham_points_y = bresenham_points[1, :].astype(np.int16)

        indGood = np.logical_and(
            np.logical_and(np.logical_and((bresenham_points_x > 1), (bresenham_points_y > 1)), (bresenham_points_x < MAP['sizex'])), (bresenham_points_y < MAP['sizey']))

        ## Update Map

        # decrease log-odds if cell observed free
        MAP['map'][bresenham_points_x[indGood], bresenham_points_y[indGood]] -= np.log(4)

        # increase log-odds if cell observed occupied
        if ((ex[i] > 1) and (ex[i] < MAP['sizex']) and (ey[i] > 1) and (ey[i] < MAP['sizey'])):
            MAP['map'][ex[i], ey[i]] += 2 * np.log(4)

    # clip range to prevent over-confidence
    MAP['map'] = np.clip(MAP['map'], -10*np.log(4), 10*np.log(4))

    # # plot original lidar points
    # plt.plot(ex, ey, '.k')
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.title("Lidar Scan")
    # plt.axis('equal')

    # # plot map
    # plt.imshow(MAP['map'], cmap='gray')
    # plt.title("Map")
    # plt.pause(0.5)

    return MAP
