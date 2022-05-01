import numpy as np
from Sensor import Sensor
# Create a new sensor, with similar properties as the original sensor,
#   but located at the given position and orientation.
def copy_sensor(sensor, pos_x, pos_y, pos_angle, detect_sigma):
    M = len(sensor.angles);
    max_range = sensor.max_range
    detect_prob = sensor.detect_prob
    
    # minimum vehicle structure
    vehicle = {'x': pos_x, 'y': pos_y, 'theta': pos_angle}

    new_sensor = Sensor(0, 2*np.pi, M, max_range, detect_sigma, detect_prob,
                        vehicle)

    return new_sensor
