'''
ECE276A WI21 PR2: Particle Filter SLAM
'''

import numpy as np

def resampling(particle_state, particle_weight, particle_count):

    particle_state_new = np.zeros((3, particle_count))
    particle_weight_new = np.tile(1 / particle_count, particle_count).reshape(1, particle_count)
    j = 0
    c = particle_weight[0, 0]

    for i in range(particle_count):
        u = np.random.uniform(0, 1/particle_count)
        beta = u + i/particle_count  # i=k-1
        while beta > c:
            j += 1
            c += particle_weight[0, j]

        # add to the new set
        particle_state_new[:, i] = particle_state[:, j]

    return particle_state_new, particle_weight_new
