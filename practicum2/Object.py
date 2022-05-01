import numpy as np

class Object:
    
    def __init__(self, pos, plot_color='b', track_id=None, ts=None):
        assert pos.shape[0] == 2  # [x,y] x n_timesteps
        assert np.all(np.isfinite(pos)), 'No nans allowed'
        self.pos = pos
        self.plot_color = plot_color  # matplotlib format
        self.track_id = track_id  # int representing the ground truth track id
        self.set_ts(list(range(pos.shape[1]))) if ts is None else self.set_ts(ts) 
 
    def set_ts(self, ts):
        ts = list(ts) if isinstance(ts, np.ndarray) else ts
        assert isinstance(ts, list)
        assert len(ts) == self.pos.shape[1]
        assert np.all(np.isfinite(ts)), 'No nans allowed'
        self.ts = ts

    def __repr__(self):
        ts = self.ts
        min_ts = min(self.ts)
        max_ts = max(self.ts)
        assert self.ts == list(range(min_ts, max_ts + 1))
        return f"Object(pos={str(self.pos.shape)}, plot_color={self.plot_color}, track_id={self.track_id}, ts=range({min_ts},{max_ts + 1}))"