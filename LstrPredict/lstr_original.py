import cv2
import numpy as np
import onnxruntime
from scipy.special import softmax
lane_colors = [(68,65,249),(44,114,243),(30,150,248),(74,132,249),(79,199,249),(109,190,144),(142, 144, 77),(161, 125, 39)]

class LSTR_orignal():

    def __init__(self, model_path):
        # Init model
        self.session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.get_input_info()
        self.get_output_info()

    def detect_lanes(self, image):
        #Get image info
        self.img_height, self.img_width, self.img_channels = image.shape

        input_tensor, mask_tensor = self.preprocess(image)

        outputs = self.predict(input_tensor, mask_tensor)
        
        detected_lanes, good_lanes = self.process_output(outputs)

        #visualization_img = self.draw_lanes(image)

        return detected_lanes, good_lanes

    def preprocess(self, img):
        # Resize
        img = cv2.resize(img,(self.input_width, self.input_height))

        # Scale input pixel values to -1 to 1
        img = cv2.normalize(img, None, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        img = img.transpose(2, 0, 1)
        input_tensor = img[np.newaxis,:,:,:].astype(np.float32)

        mask_tensor = np.zeros((1, 1, self.input_height, self.input_width), dtype=np.float32)

        return input_tensor, mask_tensor

    def predict(self, input_tensor, mask_tensor):
        # predict outputs
        outputs = self.session.run(self.output_names, {self.rgb_input_name: input_tensor, 
                                                       self.mask_input_name: mask_tensor})
        return outputs

    def process_output(self, outputs):  

        logits = outputs[0]
        pred = outputs[1]

        # Filter good lanes based on the probability
        prob = softmax(logits)
        good_detections = np.where(np.argmax(prob,axis=-1)==1)

        pred = pred[good_detections]

        lanes = []
        # Will be 3 lines in my case
        for lane_data in pred:
            bounds = lane_data[:2]
            k_2, f_2, m_2, n_1, b_2, b_3 = lane_data[2:]

            # Calculate the points for the lane
            self.num_of_points=100
            y_norm = np.linspace(bounds[0], bounds[1], num=self.num_of_points)#Higher number gives more filled line
            x_norm = (k_2 / (y_norm - f_2) ** 2 + m_2 / (y_norm - f_2) + n_1 + b_2 * y_norm - b_3)# Algo from original code
            lane_points = np.vstack((x_norm*self.img_width, y_norm*self.img_height)).astype(int)
            lanes.append(lane_points)
        self.lanes = lanes
        self.good_lanes = good_detections[1]

        return lanes, self.good_lanes

    def get_input_info(self):
        model_inputs = self.session.get_inputs()
        self.rgb_input_name = self.session.get_inputs()[0].name
        self.mask_input_name = self.session.get_inputs()[1].name

        self.input_shape = self.session.get_inputs()[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_info(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

    def draw_lanes(self,input_img):
        draw_limit=self.num_of_points/2#max points to be drawn
        # Write the detected line points in the image
        visualization_img = input_img.copy()
        for lane_num,lane_points in zip(self.good_lanes, self.lanes):
            i=0
            for lane_point in reversed(lane_points.transpose()):
                cv2.circle(visualization_img, (lane_point[0],lane_point[1]), 3, lane_colors[lane_num], -1)
                i+=1
                if i==draw_limit:#Stop drawing point if max is reached
                    break
        return visualization_img

