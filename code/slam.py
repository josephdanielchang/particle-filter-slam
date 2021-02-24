'''
ECE276A WI21 PR2: Particle Filter SLAM
'''

import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
from pr2_utils import read_data_from_csv, compute_stereo
from mapping import update_map
from prediction import prediction
from update import pose_update
from resampling import resampling
from texture_map import texture_map
import os

### Load Sensor Data

# Lidar
# rays with value 0.0 represent infinite range observations
lidar_time, lidar_data = read_data_from_csv('ECE276A_PR2_Local/sensor_data/lidar.csv')
# fov = 190  # degrees
start_angle = -5  # degrees
end_angle = 185  # degrees
angular_resolution = 0.666  # degrees
# max_range = 80  # meters

# FOG
# [timestamp, delta roll, delta pitch, delta yaw] in radians
fog_time, fog_data = read_data_from_csv('ECE276A_PR2_Local/sensor_data/fog.csv')
# fog_delta_roll = fog_data[:, 0]
# fog_delta_pitch = fog_data[:, 1]
fog_delta_yaw = fog_data[:, 2]

# Encoder
# [timestamp, left count, right count]
encoder_time, encoder_data = read_data_from_csv('ECE276A_PR2_Local/sensor_data/encoder.csv')
encoder_left_count = encoder_data[:, 0]
encoder_right_count = encoder_data[:, 1]
encoder_resolution = 4096
encoder_left_wheel_diameter = 0.623479
encoder_right_wheel_diameter = 0.622806
# encoder_wheel_base = 1.52439

# Stereo Images
stereo_time = os.listdir('ECE276A_PR2_Local/stereo_images/stereo_left')
stereo_baseline = 475.143600050775  # mm
# left camera intrinsic/camera matrix
K = [8.1690378992770002e+02, 5.0510166700000003e-01, 6.0850726281690004e+02,
     0., 8.1156803828490001e+02, 2.6347599764440002e+02,
     0., 0., 1]
K_inverse = np.linalg.inv(K)

### Transforms
### RPY(roll/pitch/yaw = XYZ extrinsic, degree), R(rotation matrix), T(translation matrix)

# Body2FOG
# RPY_fog = [0, 0, 0]
# R_fog = [1, 0, 0, 0, 1, 0, 0, 0, 1]
# T_fog = [-0.335, -0.035, 0.78]

# Body2Lidar
# RPY_lidar = [142.759, 0.0584636, 89.9254]
R_lidar = [0.00130201, 0.796097, 0.605167, 0.999999, -0.000419027, -0.00160026, -0.00102038, 0.605169, -0.796097]
T_lidar = [0.8349, -0.0126869, 1.76416]

# Lidar2Body
transform_lidar2body = np.array([[R_lidar[0], R_lidar[1], R_lidar[2], T_lidar[0]],
                                 [R_lidar[3], R_lidar[4], R_lidar[5], T_lidar[1]],
                                 [R_lidar[6], R_lidar[7], R_lidar[8], T_lidar[2]],
                                 [0, 0, 0, 1]])

# Stereo2Body
RPY_stereo = [-90.878, 0.0132, -90.3899]
R_stereo = [-0.00680499, -0.0153215, 0.99985, -0.999977, 0.000334627, -0.00680066, -0.000230383, -0.999883, -0.0153234]
T_stereo = [1.64239, 0.247401, 1.58411]
transform_camera2body = np.array([[R_stereo[0], R_stereo[1], R_stereo[2], T_stereo[0]],
                                 [R_stereo[3], R_stereo[4], R_stereo[5], T_stereo[1]],
                                 [R_stereo[6], R_stereo[7], R_stereo[8], T_stereo[2]],
                                 [0, 0, 0, 1]])

# print data sizes
lidar_length = len(lidar_time)      # (115865)
fog_length = len(fog_time)          # (1160508)
encoder_length = len(encoder_time)  # (116048)
stereo_length = len(stereo_time)    # (1161)
print('lidar_length:', lidar_length)
print('fog_length:', fog_length)
print('encoder_length:', encoder_length)
print('stereo_length:', stereo_length)

### Downsample lidar data
# lidar_data = lidar_data[::5]

### Sync FOG to encoder timestamps
# fog_delta_yaw_synced = np.zeros(len(encoder_time))
# fog_sum = 0
# index1 = 0
# for i in range(len(encoder_time)):
#     if i % 10000 == 0:
#         print(i)
#     diff = abs(fog_time - encoder_time[i])
#     index2 = np.argmin(diff)
#
#     if i != 0:
#         # sum fog data from previous to current sync
#         fog_delta_yaw_synced[i] = sum(fog_delta_yaw[index1+1:index2])
#         index1 = index2
#     else:
#         fog_delta_yaw_synced[i] = fog_delta_yaw[index2]
#         index1 = i
# np.save('ECE276A_PR2_Local/code/fog_delta_yaw.npy', fog_delta_yaw_synced)
# fog_delta_yaw = fog_delta_yaw_synced
# print('fog synced')

fog_delta_yaw = np.load('ECE276A_PR2_Local/code/fog_delta_yaw.npy')
print('fog_delta_yaw:', fog_delta_yaw.shape)

### Mapping: Initialize First Lidar Map

# initialize MAP
MAP = {}
MAP['res'] = 1  # meters
MAP['xmin'] = -100  #meters
MAP['ymin'] = -1200
MAP['xmax'] = 1300
MAP['ymax'] = 200
MAP['sizex'] = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1))  # cells along x
MAP['sizey'] = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))  # cells along y
MAP['map'] = np.zeros((MAP['sizex'], MAP['sizey']), dtype=np.float32)  # DATA TYPE: char or int8
output_texture_map = np.zeros((MAP['sizex'], MAP['sizey'], 3), dtype=np.uint8)

# initial particle set
particle_count = 40  # N
particle_state = np.zeros((3, particle_count))  # mu = [x,y,theta]
particle_weight = np.zeros((1, particle_count))  # alpha = 1/N
particle_weight[0, 0:particle_count] = 1 / particle_count

# convert lidar to angles (rad) and ranges (m)
angle_slices = int(np.ceil((end_angle-start_angle)/angular_resolution))
angles = np.linspace(start_angle, end_angle, angle_slices) / 180 * np.pi
ranges = lidar_data[0, :]

# remove lidar outside 2-75m
indValid = np.logical_and((ranges < 75), (ranges > 2))
ranges = ranges[indValid]
angles = angles[indValid]

# update map at xy lidar values
print('update map')
particle_max_position = np.argmax(particle_weight)
particle_max_state = particle_state[:, particle_max_position]
MAP = update_map(MAP, particle_max_state, ranges, angles, transform_lidar2body)

# initialize trajectories
trajectory = np.array([[0],[0]])

encoder_index = 0
lidar_index = 0
stereo_index = 0

for i in range(0, len(encoder_time) + len(lidar_time)):
    if i % 100 == 0:
        print('SLAM iteration', i)

        # recover map pmf from log-odds map
        output_map = ((1 - 1 / (1 + np.exp(MAP['map']))) < 0.1).astype(np.int)
        output_wall = ((1 - 1 / (1 + np.exp(MAP['map']))) > 0.9).astype(np.int)

        # convert end point in world frame to cells
        ex = np.ceil((trajectory[0, :] - MAP['xmin']) / MAP['res']).astype(np.int16) - 1
        ey = np.ceil((trajectory[1, :] - MAP['ymin']) / MAP['res']).astype(np.int16) - 1

        indGood = np.logical_and(np.logical_and(np.logical_and((ex > 1), (ey > 1)), (ex < MAP['sizex'])),
                                 (ey < MAP['sizey']))

        output_map[ex[indGood], ey[indGood]] = 2
        output_wall[ex[indGood], ey[indGood]] = 2
        # output_texture_map[ex[indGood], ey[indGood]] = np.array([255, 255, 0])

        # plot map
        plt.imshow(MAP['map'], cmap='gray')
        plt.title("Map")
        plt.pause(0.01)

        # save map
        if i % 10000 == 0:
            plt.imsave('ECE276A_PR2_Local/saved_map/' + str(i) + '.png', output_map, cmap='gray')
            plt.imsave('ECE276A_PR2_Local/saved_wall/' + str(i) + '.png', output_wall, cmap='gray')
            # plt.imsave('ECE276A_PR2_Local/saved_texture_map/' + str(i) + '.png', output_texture_map, cmap='gray')

    ### Prediction (when get encoder reading)

    if encoder_time[encoder_index] < lidar_time[lidar_index]:

        # use encoders and FOG data to compute instantaneous linear and angular velocities vt and wt
        tau = (encoder_time[i] - encoder_time[i-1]) / 10**9
        if encoder_index != 0:
            left_linear_velocity = (math.pi * encoder_left_wheel_diameter * (encoder_left_count[encoder_index] - encoder_left_count[encoder_index-1])) / (encoder_resolution * tau)
        else:
            left_linear_velocity = (math.pi * encoder_left_wheel_diameter * encoder_left_count[encoder_index]) / encoder_resolution
        if encoder_index != 0:
            right_linear_velocity = (math.pi * encoder_right_wheel_diameter * (encoder_right_count[encoder_index] - encoder_right_count[encoder_index-1])) / (encoder_resolution * tau)
        else:
            right_linear_velocity = (math.pi * encoder_right_wheel_diameter * encoder_right_count[encoder_index]) / encoder_resolution

        linear_velocity = (left_linear_velocity + right_linear_velocity) / 2  # m/s
        print('linear velocity:', linear_velocity)

        # predict vehicle particle pose
        particle_state = prediction(particle_state, linear_velocity, fog_delta_yaw[encoder_index], tau)

        if encoder_index < encoder_length-1:
            encoder_index += 1
        else:
            lidar_index += 1

    ### Update (when get lidar reading)

    else:

        # convert lidar to angles (rad) and ranges (m)
        angles = np.linspace(start_angle, end_angle, 286) / 180 * np.pi
        ranges = lidar_data[lidar_index, :]

        # remove lidar outside 2-75m
        indValid = np.logical_and((ranges < 75), (ranges > 2))
        ranges = ranges[indValid]
        angles = angles[indValid]

        # update particle filtering weights
        particle_state, particle_weight = pose_update(MAP, particle_state, particle_weight, ranges, angles, transform_lidar2body, particle_count)

        ### Mapping: update map at xy lidar values
        particle_max_position = np.argmax(particle_weight)
        particle_max_state = particle_state[:, particle_max_position]
        trajectory = np.hstack((trajectory, particle_max_state[0:2].reshape(2, 1)))
        MAP = update_map(MAP, particle_max_state, ranges, angles, transform_lidar2body)

        ### Texture Mapping
        if lidar_index % 100 == 0:
            stereo_index += 1

        diff = abs(stereo_time - lidar_time[lidar_index])
        index = np.argmin(diff)

        path_l = 'ECE276A_PR2_Local/stereo_images/stereo_left/%s.png' % stereo_time[stereo_index]
        path_r = 'ECE276A_PR2_Local/stereo_images/stereo_right/%s.png' % stereo_time[stereo_index]

        # disparity, image_l, image_r = compute_stereo(path_l, path_r)  # images in BGR
        #
        # output_texture_map = texture_map(MAP, image_l, disparity, transform_camera2body, particle_max_position, output_texture_map, K_inverse)

        ### Resampling

        N_eff = 1/np.dot(particle_weight.reshape(1, particle_count), particle_weight.reshape(particle_count, 1))
        if N_eff < 8:
            particle_state, particle_weight = resampling(particle_state, particle_weight, particle_count)

        if lidar_index < lidar_length:
            lidar_index += 1
        else:
            encoder_index += 1
