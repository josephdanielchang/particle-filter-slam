'''
ECE276A WI21 PR2: Particle Filter SLAM
'''

import numpy as np

## Estimate the robot trajectory via differential drive motion model

def prediction(particle_state, linear_velocity, theta_change, tau):

    # current particle pose
    x_t = particle_state[0, :]
    y_t = particle_state[1, :]
    theta_t = particle_state[2, :]

    # change in robot pose
    x_change = tau * linear_velocity * np.cos(theta_t + theta_change)
    y_change = tau * linear_velocity * np.sin(theta_t + theta_change)

    N = np.shape(particle_state)[1]

    # new particle pose and add noise
    x_t += x_change
    y_t += y_change
    theta_t += theta_change

    # gaussian noise
    x_t += np.array([np.random.normal(0, abs(np.max(x_change)) / 10, N)])[0]
    y_t += np.array([np.random.normal(0, abs(np.max(y_change)) / 10, N)])[0]
    theta_t += np.array([np.random.normal(0, abs(theta_change) / 10, N)])[0]
    print('angle:', theta_t[0])

    particle_state[0, :] = x_t
    particle_state[1, :] = y_t
    particle_state[2, :] = theta_t

    return particle_state
