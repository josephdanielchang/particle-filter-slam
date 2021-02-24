'''
ECE276A WI21 PR2: Particle Filter SLAM
'''

import numpy as np
from pr2_utils import read_data_from_csv
from mapping import update_map

### Load Sensor Data

# Lidar
# rays with value 0.0 represent infinite range observations
lidar_time, lidar_data = read_data_from_csv('ECE276A_PR2_Local/sensor_data/lidar.csv')
fov = 190  # degrees
start_angle = -5  # degrees
end_angle = 185  # degrees
angular_resolution = 0.666  # degrees
max_range = 80  # meters

# FOG
# [timestamp, delta roll, delta pitch, delta yaw] in radians
fog_time, fog_data = read_data_from_csv('ECE276A_PR2_Local/sensor_data/fog.csv')
fog_delta_roll = fog_data[:, 0]
fog_delta_pitch = fog_data[:, 1]
fog_delta_yaw = fog_data[:, 2]

# Encoder
# [timestamp, left count, right count]
encoder_time, encoder_data = read_data_from_csv('ECE276A_PR2_Local/sensor_data/encoder.csv')
encoder_left_count = encoder_data[:, 0]
encoder_right_count = encoder_data[:, 1]
encoder_resolution = 2096
encoder_left_wheel_diameter = 0.623479
encoder_right_wheel_diameter = 0.622806
encoder_wheel_base = 1.52439

### Transforms
### RPY(roll/pitch/yaw = XYZ extrinsic, degree), R(rotation matrix), T(translation matrix)

# Body2FOG
RPY_fog = [0, 0, 0]
R_fog = [1, 0, 0, 0, 1, 0, 0, 0, 1]
T_fog = [-0.335, -0.035, 0.78]

# Body2Lidar
RPY_lidar = [142.759, 0.0584636, 89.9254]
R_lidar = [0.00130201, 0.796097, 0.605167, 0.999999, -0.000419027, -0.00160026, -0.00102038, 0.605169, -0.796097]
T_lidar = [0.8349, -0.0126869, 1.76416]

# Body2Stereo
RPY_stereo = [-90.878, 0.0132, -90.3899]
R_stereo = [-0.00680499, -0.0153215, 0.99985, -0.999977, 0.000334627, -0.00680066, -0.000230383, -0.999883, -0.0153234]
T_stereo = [1.64239, 0.247401, 1.58411]

# Lidar2Body
transform_lidar2body = np.array([[R_lidar[0], R_lidar[1], R_lidar[2], T_lidar[0]],
                                 [R_lidar[3], R_lidar[4], R_lidar[5], T_lidar[1]],
                                 [R_lidar[6], R_lidar[7], R_lidar[8], T_lidar[2]],
                                 [0, 0, 0, 1]])

### Mapping: Initialize Map with First Lidar Scan

# initialize MAP
MAP = {}
MAP['res'] = 0.1  # meters
MAP['xmin'] = -50  # meters
MAP['ymin'] = -50
MAP['xmax'] = 50
MAP['ymax'] = 50
MAP['sizex'] = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1))  # cells along x
MAP['sizey'] = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))  # cells along y
MAP['map'] = np.zeros((MAP['sizex'], MAP['sizey']), dtype=np.float32)  # DATA TYPE: char or int8

# initial particle set
particle_count = 40  # N
particle_state = np.zeros((3, particle_count))  # mu = [x,y,theta]
particle_weight = np.zeros((1, particle_count))  # alpha = 1/N
particle_weight[0, 0:particle_count] = 1 / particle_count

# convert lidar to angles (rad) and ranges (m)
angles = np.linspace(start_angle, end_angle, 286) / 180 * np.pi
ranges = lidar_data[0, :]

# remove lidar outside 2-75m
indValid = np.logical_and((ranges < 75), (ranges > 2))
ranges = ranges[indValid]
angles = angles[indValid]

# convert scan to cartesian coordinates in lidar frame
xs0 = ranges * np.cos(angles)
ys0 = ranges * np.sin(angles)

# update map at xy lidar values (uncomment plotting in mapping.py for this file)
particle_max_position = np.argmax(particle_weight)
particle_max_state = particle_state[:, particle_max_position]
MAP = update_map(MAP, particle_max_state, ranges, angles, transform_lidar2body)
