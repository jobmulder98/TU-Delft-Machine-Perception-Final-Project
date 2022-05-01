import matplotlib.pyplot as plt
import numpy as np
from copy_sensor import copy_sensor
from visualizations import plot_setup_selfloc, plot_vehicle_state,\
                            plot_sensor_rays, plot_current_measurements

def compare_measurements(map_, sensors, vehicles, control_inputs, measurements, 
                         map_measurement_loglik=None, test_location_id=0):
    # let's take the first time step as reference
    sensor = sensors[0]
    vehicle = vehicles[0]
    control_input = control_inputs[0]
    measurement = measurements[0]

    test_particles = [
            [0, 10, 2*np.pi * 0/8], # first test location
            [0, 10, 2*np.pi * 2/8], # second test location
            [0, 16, 2*np.pi * 4/8], # third test location
            [0, 33, 2*np.pi * 4/8], # etc ...
            [10, 22, 2*np.pi * 2/8], 
            [-12, 22, 2*np.pi * 12/8], 
            [-10, 22, 2*np.pi * 13/8] 
    ]
    particle = test_particles[test_location_id]

    # The following lines create a hypothetical 'ideal' or 'expected' 
    #   measurement for the selected particle.
    #   This represents what we *expect* to measure if the vehicle is actually
    #   at the particle position.
    particle_sensor = copy_sensor(sensor, particle[0], particle[1], particle[2], 0.) # copy our sensor, but put it on the particle's state
    particle_meas = particle_sensor.new_meas() # create a new virtual measurement for the sensor
    particle_meas = particle_sensor.observe_point(particle_meas,\
                        list(map_['obstacles']), 1.) # measure the obstacles in the map

    if map_measurement_loglik is not None:
        log_weight = map_measurement_loglik(particle, map_, measurement, sensor)

        print(f'expected measurement at particle x_t\n'
                f'log weight = {log_weight:.4}, i.e. '
                f'weight = {np.exp(log_weight):.4}\n')

    # -- visualize --
    plt.subplots(2,2,figsize=(12,10))

    plt.subplot(1,2,1)
    plot_setup_selfloc(map_)
    ax = plt.gca()

    # true vehicle state and measurements
    plot_vehicle_state(ax, vehicle)
    plot_sensor_rays(ax, sensor)
    plot_current_measurements(ax, sensor, measurement)

    # particle's vehicle state and measurements
    plot_vehicle_state(ax, {'x': particle[0], 'y': particle[1], 
                        'theta': particle[2], 'kappa': 0})
    plot_sensor_rays(ax, particle_sensor)
    plot_current_measurements(ax, particle_sensor, particle_meas)

    # show the 'ideal' sensor measurement
    plt.subplot(2,2,2)
    ax = plt.gca()
    ax.plot(sensor.angles, measurement.dists, c='r')
    ax.set_ylim([0, 30])
    ax.set_ylabel('distance (m)')
    ax.set_title('actual measurement z_t')

    # also show the 'expected' measurement at the particle's position,orientation
    plt.subplot(2,2,4)
    ax = plt.gca()
    ax.plot(particle_meas.dists, c='b')
    ax.set_ylim([0, 30])
    ax.set_xlabel('sensor ray')
    ax.set_ylabel('distance (m)')
    ax.set_title('expected measurement at particle x_t')

    plt.pause(.1)
