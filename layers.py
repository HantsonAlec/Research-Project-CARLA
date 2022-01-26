import carla
import argparse

argparser = argparse.ArgumentParser(
    description='CARLA Manual Control Client')
argparser.add_argument(
    '-p', '--port',
    metavar='P',
    default=2000,
    type=int,
    help='TCP port to listen to (default: 2000)')
args = argparser.parse_args()

client = carla.Client('localhost', args.port)
client.set_timeout(5.0)
world = client.load_world(
    'Town02_Opt', carla.MapLayer.Buildings | carla.MapLayer.ParkedVehicles)

# Toggle all buildings off
# world.unload_map_layer(carla.MapLayer.Buildings)
