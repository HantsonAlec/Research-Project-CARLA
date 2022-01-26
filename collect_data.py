import glob
import os
import sys
import random
import numpy as np
import time
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================


import carla

IM_WIDTH = 640
IM_HEIGHT = 480
actor_list = []


def process_img(image):
    i = np.array(image.raw_data)
    print(i.shape)
    i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
    i3 = i2[:, :, :3]
    return i3/255.0


try:
    # Connect
    client = carla.Client("localhost", 2000)
    client.set_timeout(5.0)

    # get world
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    # pick car
    bp = blueprint_library.filter("model3")[0]

    # Spawn car
    spawn_point = random.choice(world.get_map().get_spawn_points())
    vehicle = world.spawn_actor(bp,  spawn_point)

    # Autopilot
    vehicle.set_autopilot(True)
    # Manual
    #vehicle.apply_control(carla.VehileControl(throttle=1.0, steer=0.0))

    # append spwaned vehicle to actors
    actor_list.append(vehicle)

    # Make camera bp
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute("image_size_x", f"{IM_WIDTH}")
    camera_bp.set_attribute("image_size_y", f"{IM_HEIGHT}")

    # Place bp on car
    spawn_point = carla.Transform(
        carla.Location(x=1, z=1.7))  # Camera location
    sensor = world.spawn_actor(camera_bp, spawn_point, attach_to=vehicle)
    actor_list.append(sensor)

    # Get sensor data
    sensor.listen(lambda image: image.save_to_disk(
        'output/%06d.png' % image.frame_number))

    # Alternative convert image to array
    #sensor.listen(lambda image: process_img(image))
    time.sleep(60)
finally:
    # cleanup
    for actor in actor_list:
        actor.destroy()
    print('All cleaned up')
