import copy
import math

import numpy as np


# ========== Vehicle state == #
# -- define the vehicle dynamics --
# See Algorithm 3 [Ferguson, 2008]
# state: [x, y, theta, kappa, v, a]
#   x     : vehicle positional x
#   y     : vehicle positional y
#   theta : vehicle orientation
#   kappa : curvature
#   v     : velocity
#   a     : acceleration
# control:
#   u_acc   : acceleration command
#   u_steer : steering angle command
class vehicle_state:
    def __init__(self, x, y, theta, kappa, v, a):
        self.x = x
        self.y = y
        self.theta = theta
        self.kappa = kappa
        self.v = v
        self.a = a

    def motion_model(self, u_vel, u_steer, dt):
        self.x += self.v * np.sin(self.theta) * dt
        self.y += self.v * np.cos(self.theta) * dt
        self.theta += self.v * self.kappa * dt
        self.kappa = u_steer
        # self.a = u_acc
        self.v = u_vel  # self.a * dt

        # No backwards driving
        if self.v <= 0:
            self.v = 0

    def get_goal_state(self, road, length, lat_offset, init_state):
        target_x, target_y, target_theta = make_road_xy(
            road.rradius, road.init_y + road.rlength * length, road.init_x + lat_offset
        )

        new_state = copy.deepcopy(init_state)
        new_state.x = target_x
        new_state.y = target_y
        new_state.theta = target_theta
        return new_state


class vehicle:
    def __init__(self, state_vec, prev_state_vec, dt):

        self.state = vehicle_state(
            state_vec[0],
            state_vec[1],
            math.atan2(state_vec[1] - prev_state_vec[1], state_vec[0] - prev_state_vec[0])
            - np.pi / 2,  # Direction of movement
            0.0,
            math.sqrt((state_vec[1] - prev_state_vec[1]) ** 2 + (state_vec[0] - prev_state_vec[0]) ** 2) / dt,
            0.0,
        )

    def __str__(self):
        return (
            "Vehicle at ("
            + str(self.state.x)
            + ", "
            + str(self.state.y)
            + "),\ttheta = "
            + str(self.state.theta)
            + ",\tv = "
            + str(self.state.v)
        )


# ========== Road =========== #
# Define the road/lane following scenario
# Input:
#  - rradius : road curvature radius (meter), rradius --> inf = straight
#  - rlength : length of road segment (meter)
#  - rwidth  : width of road / lane (meter)
#  - rsteps  : discretization steps of road representation
# Output:
#  - road : struct containing the x,y points of the different lines
class road:
    def __init__(self, vehicle_state, radius=50, length=40, width=4, steps=100):

        self.rradius = radius
        self.rlength = length
        self.rwidth = width
        self.rsteps = steps
        self.init_x = vehicle_state.x
        self.init_y = vehicle_state.y #- 10.0

        rlong = np.linspace(
            0.0, self.rlength, self.rsteps
        )  # np.linspace(0, self.rlength, self.rsteps)  # distance in meters along road

        # compute world-coordinates of road left, center and right border
        [rx, ry, rtheta] = make_road_xy(
            self.rradius,
            self.init_y + rlong,
            self.init_x + np.array([0, -self.rwidth / 2, +self.rwidth / 2, -self.rwidth * 1.5]),
        )

        # Set class variables
        self.rlong = rlong

        self.rx = rx
        self.ry = ry
        self.rtheta = rtheta


def make_road_xy(rradius, rlong, rlat=np.array([0])):

    if np.isinf(rradius):
        # special case
        L = len(rlong)
        D = len(rlat)

        rtheta = np.zeros([1, L])
        rx = rlat * np.ones([1, L])
        ry = np.ones([D, 1]) * rlong
        return np.squeeze(rx), np.squeeze(ry), np.squeeze(rtheta)

    # compute road center line
    rtheta = rlong / rradius

    # Make sure both variables have 2 dimensions
    rlat = np.expand_dims(np.atleast_1d(rlat), axis=1)
    rtheta = np.expand_dims(np.atleast_1d(rtheta), axis=0)

    rx = (-rradius + rlat) * (np.cos(-rtheta))
    ry = (-rradius + rlat) * (np.sin(-rtheta))

    # let (0,0) not be center of turn, but start point of curve
    rx = rx + rradius

    return np.squeeze(rx), np.squeeze(ry), np.squeeze(rtheta)


# ===== Dynamic Obstacles ===== #
class dynamic_obstacle_state:
    def __init__(self, state_vec):
        self.x = state_vec[0]
        self.y = state_vec[1]
        self.v = math.sqrt(state_vec[2] ** 2 + state_vec[3] ** 2)

        if self.v > 0:
            self.theta = math.atan2(state_vec[3], state_vec[2])  # Find the orientation by vy / vx
        else:
            self.theta = 0.0


# Possibly not an interface but class for holding predictions
class dynamic_obstacle:
    def __init__(self, state_vec, track_index):
        self.state = dynamic_obstacle_state(state_vec)
        self.track_index = track_index
        self.prediction = self.predict_motion(100, 0.1)  # Todo
        self.plot_color = None

    def __str__(self):
        return (
            "Pedestrian at ("
            + str(self.state.x)
            + ", "
            + str(self.state.y)
            + "),\ttheta = "
            + str(self.state.theta)
            + ",\tv = "
            + str(self.state.v)
        )

    # Constant velocity prediction of the pedestrian motion
    def predict_motion(self, N, dt):
        prediction = np.zeros((N, 2))
        for idx in range(N):

            # For the first stage we use the detected pedestrian location
            if idx is 0:
                prediction[idx, 0] = self.state.x + self.state.v * math.cos(self.state.theta) * dt
                prediction[idx, 1] = self.state.y + self.state.v * math.sin(self.state.theta) * dt
            else:
                prediction[idx, 0] = prediction[idx - 1, 0] + self.state.v * math.cos(self.state.theta) * dt
                prediction[idx, 1] = prediction[idx - 1, 1] + self.state.v * math.sin(self.state.theta) * dt

        return prediction


# Candidates
# A class representing a candidate trajectory
class candidate:
    def __init__(self, lat_offset, length, params, states):
        self.lat_offset = lat_offset
        self.long_offset = length
        self.params = params
        self.states = states


def find_best_candidate(candidates):
    # Retrieve the cost of each candidate
    candidate_cost = list(c.total_cost for c in candidates)

    # determine optimal candidate and its index
    best_cost = min(candidate_cost)
    best_cidx = np.where(candidate_cost == best_cost)[0]

    if best_cidx.size > 0:
        cidx = best_cidx[0]
    else:
        cidx = 0
        print("Warning: No best candidate trajectory found (or all trajectories are of equal cost)!")

    s_candidate = candidates[cidx]
    states = s_candidate.states

    return s_candidate, states, candidate_cost, cidx


def is_velocity_profile_implemented(test_profile, init_state):
    # We check if you implemented the velocity profile (only used to define the candidates)
    test_frac = np.linspace(0, 1, 25)
    return not (
        min(test_profile.evaluate(test_frac)) == init_state.v and max(test_profile.evaluate(test_frac)) == init_state.v
    )
