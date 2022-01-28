import cv2
import numpy as np
import warnings
warnings.filterwarnings("ignore")


class HoughTransform():
    def __init__(self):
        pass

    def detect_lanes(self, image):
        # preprocess
        isolated = self.preprocess(image)
        # Get Hough lines
        lines = cv2.HoughLinesP(
            isolated, 2, np.pi/180, 100, np.array([]), minLineLength=30, maxLineGap=5)
        # Avergae the lines to get the points
        averaged_lines = self.average(image, lines)
        return averaged_lines

    def preprocess(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 50)
        isolated = self.region(edges)
        return isolated

    def region(self, edges):
        height, width = edges.shape
        voi = np.array([
                       # View of interest in traingle shape
                       [(10, height), (int(width/2), int(height/2)), (width, height)]
                       ])
        mask = np.zeros_like(edges)  # make mask size of image(edges)
        mask = cv2.fillPoly(mask, voi, 255)
        mask = cv2.bitwise_and(edges, mask)  # Keep only what is in VOI
        return mask

    def average(self, image, lines):
        left_points = []
        right_points = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line.reshape(4)  # Coordinates
                parameters = np.polyfit((x1, x2), (y1, y2), 1)

                slope = parameters[0]
                y_int = parameters[1]
                slope = round(slope, 16)
                if slope < 0:
                    left_points.append((slope, y_int))
                elif slope > 0:
                    right_points.append((slope, y_int))

        # takes average
        right_avg = np.average(right_points, axis=0)
        left_avg = np.average(left_points, axis=0)

        left_line = self.make_points(image, left_avg)
        right_line = self.make_points(image, right_avg)
        return np.array([left_line, right_line])

    def make_points(self, image, average):
        if isinstance(average, np.ndarray):
            slope, y_int = average
            y1 = image.shape[0]
            y2 = int(y1 * (3/5))  # 3/5 determines length of the lines
            if -5 <= slope <= 5:
                x1 = int((y1 - y_int) // slope)
                x2 = int((y2 - y_int) // slope)
            else:
                x1 = x2 = 0
        else:
            y1 = y2 = x1 = x2 = 0
        return np.array([x1, y1, x2, y2])
