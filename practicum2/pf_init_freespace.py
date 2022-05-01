import numpy as np
from numpy.random import rand, randint

# Generate N 
def pf_init_freespace(N, map_):
    particles = np.zeros((3,N))
    rand_idx = randint(0, map_['freespace'].shape[0], size=N)
    particles[:2,:] = np.swapaxes(map_['freespace'][rand_idx,:],0,1)
    particles[:2,:] = particles[:2,:] + (map_['dimensions'][:,None]*(rand(2,N)-.5))
    particles[2,:] = rand(N) * 2.*np.pi
    assert particles.shape == (3,N)
    return particles
