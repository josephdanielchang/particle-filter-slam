'''
ECE276A WI21 PR2: Particle Filter SLAM
'''

import numpy as np
from pr2_utils import bresenham2D
import matplotlib.pyplot as plt

# Update log-odds map according to lidar scan

def texture_map(MAP, image_l, disparity, transform_camera2body, particle_max_position, output_texture_map, K_inverse):



    return output_texture_map