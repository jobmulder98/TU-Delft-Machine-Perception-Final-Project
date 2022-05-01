import numpy as np
from numpy.random import rand

def pf_init_around_state(N, x, y, theta):
    # init at true location
    particles = np.array([x, y, theta])
    particles = np.tile(particles.reshape(3,1), (1,N))
    assert particles.shape == (3,N)
    
    # add a bit of variation
    particles[:2,:] = particles[:2,:] + (rand(2,N)-.5)
    particles[2,:] = particles[2,:] + rand(1, N) * 2*np.pi / 16.

    return particles
