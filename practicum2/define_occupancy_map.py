import numpy as np

def define_occupancy_map():
    #frac_to_remove = 0.

    # define grid dimensions
    dim = 32
    xs = np.linspace(-25, 25, dim)
    ys = np.linspace(-5, 45, dim)
    dimensions = np.array([xs[1] - xs[0], ys[1] - ys[0]])
    X, Y = np.meshgrid(xs, ys)

    # 'draw' in the grid where walls/roads are located
    grid = np.ones((dim, dim))
    grid[(X >= -1) & (X < 2)] = 0
    grid[(Y >= 20) & (Y <= 25)] = 0
    grid[(X >= -15) & (X <= -10) & (Y < 20)] = 0
    grid[(X <= -15) & (Y >= 7) & (Y <= 12)] = 0

    # freespace is where ther are no roads
    freespace = np.concatenate([np.expand_dims(X[grid==0],axis=-1), 
                                np.expand_dims(Y[grid==0],axis=-1)], 
                                axis=-1)

    obstacles = np.concatenate([np.expand_dims(X[grid>0],axis=-1), 
                                np.expand_dims(Y[grid>0],axis=-1)], 
                                axis=-1)
    #obstacles = np.concatenate([np.expand_dims(-22.+Y[grid>0],axis=-1), 
    #                            np.expand_dims(10.+X[grid>0],axis=-1)], 
    #                            axis=-1)
    #obstacles = np.array([X[grid > 0], Y[grid > 0]]).T
    assert len(obstacles.shape) == 2

    map_ = {
            'xs' : xs,
            'ys' : ys,
            'dimensions': dimensions,
            'grid' : grid,
            'freespace' : freespace,
            'obstacles' : obstacles
            }

    return map_
