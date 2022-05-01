import matplotlib.pyplot as plt
import numpy as np
from interfaces import find_best_candidate
from matplotlib import animation, colors


def plot_scene(ax, vehicle, pedestrians, road, dynamic=False):
    ax.set_aspect("equal", adjustable="box")

    plot_setup_trajectory_planning(ax, road)

    # Scene limits
    xmin = 1e6
    xmax = -1e6
    ymax = -1e6
    for ped in pedestrians:
        xmin = min(xmin, ped.state.x)
        xmax = max(xmax, ped.state.x)
        ymax = max(ymax, ped.state.y)

    xmin = min(xmin, min(road.rx[-1, :]))
    xmax = max(xmax, max(road.rx[0, :]))
    xmax = max(vehicle.state.x, xmax)
    ymax = max(ymax, max(road.ry[-1, :]))

    xmin -= 5.0
    xmax += 5.0
    ymax += 3.0
    ymin = vehicle.state.y - 5.0

    ax.set_xlim((xmin, xmax))
    ax.set_ylim((ymin, ymax))
    ax.grid(True, alpha=0.5)

    ax.set_xlabel("lateral - x (meters)")
    ax.set_ylabel("longitudinal - y (meters)")

    plot_pedestrians(ax, pedestrians, 0, initial=True)

    if not dynamic:
        plot_vehicle_state(ax, vehicle.state)


def plot_pedestrians(ax, pedestrians, t, initial=False):
    radius = 1.0
    obj = []

    for pedestrian in pedestrians:

        posxy = [pedestrian.state.x, pedestrian.state.y]
        if np.all(np.isfinite(posxy)):
            obj.append(ax.text(posxy[0] + 0.5, posxy[1] + 0.5, str(pedestrian.track_index)))

        if initial:
            alphas = np.linspace(0.1, 0.3, len(pedestrian.prediction[t:, 0]))
            cmap = colors.LinearSegmentedColormap.from_list("incr_alpha", [(0, (*colors.to_rgb("C0"), 0)), (1, "C0")])
            obj.append(ax.scatter(pedestrian.prediction[t:, 0], pedestrian.prediction[t:, 1], 70))
            if t is 0:
                ped_state = (pedestrian.state.x, pedestrian.state.y)
            else:
                ped_state = pedestrian.prediction[t, :]

            obj.append(ax.plot(ped_state[0], ped_state[1], marker=".", markersize=20, color="black")[0])

            # obj.append(
            #     ax.plot(pedestrian.prediction[t:, 0], pedestrian.prediction[t:, 1], marker=".", markersize=18, alpha=)[0]
            # )
            # pedestrian.plot_color = obj[-1].get_color()
        else:
            # obj.append(ax.plot(pedestrian.prediction[t:, 0], pedestrian.prediction[t:, 1], marker='.', color=pedestrian.plot_color, markersize=18)[0])

            if t is 0:
                ped_state = (pedestrian.state.x, pedestrian.state.y)
            else:
                ped_state = pedestrian.prediction[t, :]

            obj.append(ax.plot(ped_state[0], ped_state[1], marker=".", markersize=20, color="black")[0])

            circle = plt.Circle(ped_state, radius, alpha=0.3)
            collision_obj = ax.add_patch(circle)
            obj.append(collision_obj)

    obj[-3].set_label("Pedestrian")
    obj[-2].set_label("Predicted Pedestrian Path")

    return obj


def animate_final_trajectory(vehicle, pedestrians, scene_road, candidates):
    # Two plots: static scene and the animation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 10))

    plot_scene(ax1, vehicle, pedestrians, scene_road)

    s_candidate, states, candidate_cost, cidx = find_best_candidate(candidates)

    # Compute the range of costs for coloring
    # In case infinite costs occur, resolve
    candidate_cost = np.array(candidate_cost)
    cost_range_min = min(candidate_cost[candidate_cost != np.inf])
    cost_range_max = max(candidate_cost[candidate_cost != np.inf])
    cost_range = cost_range_max - cost_range_min
    candidate_cost[candidate_cost == np.inf] = cost_range_max

    # for c in candidates:
    #     total_cost = c.total_cost
    #     states = c.states
    #
    #     # use the cost to visualize good (green) and bad (red) trajectories
    #     alpha = float((total_cost - cost_range_min) / (cost_range + 1e-5))
    #     plot_trajectory(ax1, states, color=(alpha, 1.0 - alpha, 0.0))

    # Plot all candidates and compute their cost
    for c in candidates:
        total_cost = min(c.total_cost, cost_range_max)
        candidate_states = c.states

        # use the to visualize good (green) and bad (red) trajectories
        alpha = float((total_cost - cost_range_min) / (cost_range + 1e-5))  # per element
        if c == candidates[0]:
            plot_trajectory(
                ax1, candidate_states, color=(alpha, 1.0 - alpha, 0.0), linewidth=0.5, label="Candidate Trajectory"
            )
        else:
            plot_trajectory(ax1, candidate_states, color=(alpha, 1.0 - alpha, 0.0), linewidth=0.5)

    ax1.legend(loc="lower left")

    print("Selected candidate %d [Cost = %.3f]" % (cidx, s_candidate.total_cost))

    plot_scene(ax2, vehicle, pedestrians, scene_road, dynamic=True)
    plot_trajectory_cost(ax2, s_candidate.states, s_candidate.cost_per_timestep)
    print("Plotting Candidates and the Selected Trajectory")

    plt.show()

    print("Creating Animation... (this may take a while!)")
    n_timesteps = min(s_candidate.states.size, pedestrians[0].prediction.shape[0])

    ims = []
    for t in range(n_timesteps):
        objs = []
        state = s_candidate.states[t]
        objs = animate_vehicle_trajectory(ax2, state)
        objs.extend(plot_pedestrians(ax2, pedestrians, t, initial=False))

        ims.append(objs)

    anim = animation.ArtistAnimation(fig, ims, repeat=False, interval=100)
    plt.close(fig)
    return anim, candidate_cost, s_candidate, states


# Plot functionality
def box_in_frame(ax, cx, cy, w, h, R, T, **kwargs):
    # car outset
    points = np.array([[1, -1, -1, 1, 1], [-1, -1, 1, 1, -1]]).astype(float)

    points[0, :] = points[0, :] * (float(w) / 2.0) + cx
    points[1, :] = points[1, :] * (float(h) / 2.0) + cy
    artist_obj = plot_in_frame(ax, points, R, T, **kwargs)

    return artist_obj


def plot_in_frame(ax, points, R, T, **kwargs):
    # Apply transformation
    points = R.dot(points)

    (artist_obj,) = ax.plot(points[0, :] + T[0], points[1, :] + T[1], **kwargs)
    return artist_obj


def plot_vehicle_state(ax, s, **kwargs):
    artist_obj_list = []

    # rotation, translation
    ct = np.cos(s.theta)
    st = np.sin(s.theta)
    R = np.array([[ct, st], [-st, ct]])
    T = np.array([s.x, s.y])

    slong = 4.5  # longitudinal size(meter)
    slat = 2.0  # lateral size(meter)

    if "color" in kwargs:
        artist_obj_list.append(box_in_frame(ax, 0.0, 0.0, slat, slong, R, T, **kwargs))  # car outset
        artist_obj_list.append(
            box_in_frame(ax, 0.0, slong * 0.05, slat * 0.8, slong * 0.2, R, T, **kwargs)
        )  # front windshield
        artist_obj_list.append(
            box_in_frame(ax, 0.0, slong * -0.25, slat * 0.8, slong * 0.15, R, T, **kwargs)
        )  # back window
    else:
        artist_obj_list.append(box_in_frame(ax, 0.0, 0.0, slat, slong, R, T, color="black", **kwargs))  # car outset
        artist_obj_list.append(
            box_in_frame(ax, 0.0, slong * 0.05, slat * 0.8, slong * 0.2, R, T, color="black", **kwargs)
        )  # front windshield
        artist_obj_list.append(
            box_in_frame(ax, 0.0, slong * -0.25, slat * 0.8, slong * 0.15, R, T, color="black", **kwargs)
        )  # back window

    # wheel angle
    kappa_mult = 1
    kct = np.cos(s.kappa * kappa_mult)
    kst = np.sin(s.kappa * kappa_mult)
    kR = np.array([[kct, kst], [-kst, kct]])

    points = np.array([[0.0, 0.0], np.array([-0.2, 0.2]) * slong])

    points_left = kR.dot(points) + np.array([[-0.35 * slat, 0.3 * slong], [-0.35 * slat, 0.3 * slong]]).transpose()
    points_right = kR.dot(points) + np.array([[0.35 * slat, 0.3 * slong], [0.35 * slat, 0.3 * slong]]).transpose()

    if "color" in kwargs:
        artist_obj_list.append(plot_in_frame(ax, points_left, R, T, linewidth=2, **kwargs))
        artist_obj_list.append(plot_in_frame(ax, points_right, R, T, linewidth=2, **kwargs))
    else:
        artist_obj_list.append(plot_in_frame(ax, points_left, R, T, color="red", linewidth=2, **kwargs))
        artist_obj_list.append(plot_in_frame(ax, points_right, R, T, color="red", linewidth=2, **kwargs))

    return artist_obj_list


# Visualization
def plot_setup_trajectory_planning(ax, road):

    # fig, ax = plt.subplots(figsize=(5, 5))
    # ax = fig.gca()
    # ax.set_aspect('equal', adjustable='box')

    rx = road.rx
    ry = road.ry

    # plot the road segments
    ax.plot(rx[0, :], ry[0, :], "k--", linewidth=2, label="Lane Center")  # center
    ax.plot(rx[1, :], ry[1, :], "k-")  # left side of our lane (e.g. lane seperation)
    ax.plot(rx[2, :], ry[2, :], "k-")  # right side of our lane
    ax.plot(rx[3, :], ry[3, :], "k-")  # side of other lane (even more left)
    plt.xlabel("world x (meter)")
    plt.ylabel("world y (meter)")

    plt.xlim(-25, 25)
    plt.ylim(0, 40)
    plt.grid()

    return ax


def animate_vehicle_trajectory(ax, state):
    obj = plot_vehicle_state(ax, state)
    return obj


def plot_trajectory(ax, states, **kwargs):
    state_x = list(o.x for o in states)
    state_y = list(o.y for o in states)
    ax.plot(state_x, state_y, **kwargs)
    ax.plot(state_x[-1], state_y[-1], marker="o", markersize=3, color="black", alpha=0.8)


def plot_trajectory_cost(ax, states, cost, **kwargs):
    state_x = list(o.x for o in states)
    state_y = list(o.y for o in states)
    scatter = ax.scatter(state_x, state_y, 35, -cost, **kwargs)  # -cost for inverted colors
    legend1 = ax.legend(*scatter.legend_elements(num=3), loc="lower left", title="Cost")
    ax.add_artist(legend1)


def plot_ped_positions_per_timestep(timestep, pedestrian_states, vehicle_states, vehicle, n_tracks, dt):
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_aspect("equal", adjustable="box")

    xmin = np.nanmin(pedestrian_states[:, :, 0]) - 3.0
    xmax = np.nanmax(pedestrian_states[:, :, 0]) + 3.0
    ymin = np.nanmin(vehicle_states[:, 1]) - 3.0
    # ymax = np.nanmax(pedestrian_states[:, :, 1]) + 3.0
    ymax = np.nanmax(vehicle_states[:, 1]) + 3.

    ax.set_xlim((xmin, xmax))
    ax.set_ylim((ymin, ymax))
    ax.grid(True, alpha=0.5)

    ax.set_xlabel("lateral - x (meters)")
    ax.set_ylabel("longitudinal - y (meters)")

    for track_index in range(n_tracks):
        if track_index == 1:
            ax.plot(
                pedestrian_states[: timestep + 1, track_index, 0],
                pedestrian_states[: timestep + 1, track_index, 1],
                marker=".",
                markersize=18,
                label="Pedestrian Track",
            )
        else:
            ax.plot(
                pedestrian_states[: timestep + 1, track_index, 0],
                pedestrian_states[: timestep + 1, track_index, 1],
                marker=".",
                markersize=18,
            )
        posxy = pedestrian_states[timestep, track_index, 0:2]
        if np.all(np.isfinite(posxy)):
            ax.text(posxy[0], posxy[1], str(track_index))

    cur_vehicle = vehicle(vehicle_states[timestep, :], vehicle_states[timestep - 1, :], dt)
    plot_vehicle_state(ax, cur_vehicle.state)
    ax.legend()
