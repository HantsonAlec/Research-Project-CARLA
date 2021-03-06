import glob
import os
import sys
import argparse
import copy
from yolo.yolov5.yolov5 import YoloV5
from multiprocessing import Pool
from LstrPredict import LSTRPredict
from HoughTransform import HoughTransform
from detrCustom.detrCustom import DETR_CUSTOM
from SegFormer.SegFormer import SegFormer
from PIL import Image

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import random

try:
    import pygame
except ImportError:
    raise RuntimeError(
        'cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError(
        'cannot import numpy, make sure numpy package is installed')

try:
    import queue
except ImportError:
    import Queue as queue
import cv2
IM_WIDTH, IM_HEIGHT = 640, 480

# CarlaSyncMode is used from other carla examples


class CarlaSyncMode(object):
    """
    Context manager to synchronize output from different sensors. Synchronous
    mode is enabled as long as we are inside this context

        with CarlaSyncMode(world, sensors) as sync_mode:
            while True:
                data = sync_mode.tick(timeout=1.0)

    """

    def __init__(self, world, *sensors, **kwargs):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get('fps', 20)
        self._queues = []
        self._settings = None

    def __enter__(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds))

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)
        return self

    def tick(self, timeout):
        self.frame = self.world.tick()
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        return data

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data
    # ---------------


def draw_image_ht(image_surface, lines):
    if lines is not None:
        for line in lines:
            if min(line) < 0 or max(line) > 10000:
                line = [0, 0, 0, 0]
            x1, y1, x2, y2 = line
            # draw lines on the mask
            pygame.draw.line(image_surface, (255, 0, 0),
                             (x1, y1), (x2, y2), 10)


def draw_image_lstr(image_surface, lane_points):
    if lane_points == []:
        lane_points = [[0, 0]]
        print(lane_points)
    try:
        # print(lane_points)
        for lane in lane_points:
            for points in lane:
                pygame.draw.circle(
                    image_surface, (255, 0, 0), tuple(points), 3)
    except:
        pass


def draw_boxes_yolo(array, image_surface, labels, cord, classes, colors, font):
    #array = array[:, :, ::-1]
    n = len(labels)
    x_shape, y_shape = array.shape[1], array.shape[0]
    for i in range(n):
        row = cord[i]
        x1 = int(row[0]*x_shape)
        y1 = int(row[1]*y_shape)
        x2 = int(row[2]*x_shape)
        y2 = int(row[3]*y_shape)
        width = x2-x1
        height = y2-y1
        text_surface = font.render(classes[int(labels[i])], True, (colors[int(
            labels[i])][0], colors[int(labels[i])][1], colors[int(labels[i])][2]))
        # Plot the boxes
        pygame.draw.rect(
            image_surface, colors[int(labels[i])], (x1, y1, width, height), 2)
        image_surface.blit(text_surface, (x1, y1))


def draw_boxes_detrC(image_surface, prob, boxes, classes, colors, font):
    for p, (x1, y1, x2, y2) in zip(prob, boxes.tolist()):
        cl = p.argmax()
        width = x2-x1
        height = y2-y1
        text_surface = font.render(classes[str(cl.item())], True, (colors[int(
            cl.item())][0], colors[int(cl.item())][1], colors[int(cl.item())][2]))
        pygame.draw.rect(
            image_surface, colors[int(cl.item())], (x1, y1, width, height), 2)
        image_surface.blit(text_surface, (x1, y1))


def draw_mask(array, seg_mask):
    img = array * 0.5 + seg_mask * 0.5
    image_surface = pygame.surfarray.make_surface(img.swapaxes(0, 1))
    return image_surface


def get_image_as_array(image):
    # Make carla frame to numpy and convert from BGR to RGB
    array = np.array(image.raw_data)
    array = array.reshape((IM_HEIGHT, IM_WIDTH, 4))
    array = array[:, :, :3]
    array_converted = array[:, :, ::-1]
    array2 = copy.deepcopy(array)
    array_converted2 = copy.deepcopy(array_converted)
    return array2, array_converted2


def get_font():
    fonts = [x for x in pygame.font.get_fonts()]
    default_font = 'ubuntumono'
    font = default_font if default_font in fonts else fonts[0]
    font = pygame.font.match_font(font)
    return pygame.font.Font(font, 14)


def should_quit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                return True
    return False


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-l', '--lane',
        metavar='L',
        default='',
        type=str,
        help='Type of lanedetection to use')
    argparser.add_argument(
        '-o', '--object',
        metavar='O',
        default='',
        type=str,
        help='Type of object detection to use')
    argparser.add_argument(
        '-s', '--segmentation',
        metavar='S',
        default='',
        type=str,
        help='Type of segmentation to use')
    args = argparser.parse_args()

    # Model paths
    yolo_model_path = "./models/yolov5.pt"
    detr_model_path = "./models/epoch=148-step=119199.ckpt"

    # Initialize lane detection model
    if args.lane == 'lstr':
        lane_detector = LSTRPredict()
    elif args.lane == 'ht':
        lane_detector = HoughTransform()

    # Initialize object detection model
    if args.object == 'yolo':
        confidence = 0.7
        object_detection = YoloV5(yolo_model_path, confidence)
        classes, colors = object_detection.get_attributes()
    elif args.object == 'detr':
        confidence = 0.5
        object_detection = DETR_CUSTOM(detr_model_path, confidence)
        classes, colors = object_detection.get_attributes()

    if args.segmentation == 'segform':
        segmentator = SegFormer()

    actor_list = []
    pygame.init()

    display = pygame.display.set_mode(
        (IM_WIDTH, IM_HEIGHT),
        pygame.HWSURFACE | pygame.DOUBLEBUF)
    font = get_font()
    clock = pygame.time.Clock()

    client = carla.Client('localhost', 2000)
    client.set_timeout(5.0)

    world = client.get_world()

    blueprint_library = world.get_blueprint_library()
    # pick car
    bp = blueprint_library.filter("model3")[0]

    # Spawn car
    spawn_point = random.choice(world.get_map().get_spawn_points())
    vehicle = world.spawn_actor(bp,  spawn_point)

    # Autopilot
    vehicle.set_autopilot(True)

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

    try:
        pool = Pool(processes=5)
        # Create a synchronous mode context.
        with CarlaSyncMode(world, sensor, fps=20) as sync_mode:
            i = 0
            points = []
            while True:
                if should_quit():
                    return
                clock.tick()

                # Advance the simulation and wait for the data.
                snapshot, image_rgb = sync_mode.tick(timeout=1.0)
                # Draw the display.
                buffer, buffer_converted = get_image_as_array(image_rgb)
                image_surface = pygame.surfarray.make_surface(
                    buffer_converted.swapaxes(0, 1))

                if(args.lane == 'ht'):
                    # Don't predcit avery frame
                    if i % 20 == 0:
                        lines = lane_detector.detect_lanes(buffer)
                    # Draw
                    draw_image_ht(image_surface, lines)
                elif args.lane == 'lstr':
                    # if i % 1 == 0:
                    lane_points = lane_detector.detect_lanes(
                        buffer_converted)
                    points = lane_points if lane_points != [] else points
                    # Draw
                    draw_image_lstr(image_surface, points)

                if args.object == 'yolo':
                    labels, cord = object_detection.detect_objects(
                        buffer_converted)
                    # Draw
                    draw_boxes_yolo(buffer, image_surface,
                                    labels, cord, classes, colors, font)
                elif args.object == 'detr':
                    outputs = object_detection.predict_outputs(
                        buffer_converted)
                    probas, bboxes_scaled = object_detection.detect_objects(
                        buffer_converted, outputs)
                    # Draw
                    draw_boxes_detrC(image_surface, probas,
                                     bboxes_scaled, classes, colors, font)
                if args.segmentation == 'segform':
                    image = Image.fromarray(np.uint8(buffer_converted))
                    seg_mask = segmentator.panoptic_detection(image)
                    # Create new mask
                    image_surface = draw_mask(buffer, seg_mask)

                display.blit(image_surface, (0, 0))
                pygame.display.flip()
                i += 1

    finally:
        # time.sleep(5)
        print('destroying actors.')
        for actor in actor_list:
            actor.destroy()
        pygame.quit()
        pool.close()
        print('done.')


if __name__ == '__main__':

    try:

        main()

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
