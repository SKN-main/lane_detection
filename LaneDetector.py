import cv2
import numpy as np
import time
from Queue import Queue
from enum import Enum
from scipy.optimize import curve_fit
import copy


class Direction(Enum):
    STRAIGHT = 0
    LEFT = -1
    RIGHT = 1


class Tracker:
    def __init__(self, position, width):
        self.position = position
        self.width = width
        self.x1 = self.position[0] - self.width//2
        self.x2 = self.position[0] + self.width//2
        self.y = self.position[1]
        self.pointer_x = position[0]
        self.is_active = True
        self.deviation = 0

    def draw(self, image):
        color = None
        if self.is_active:
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)

        cv2.putText(image, self.get_direction().name + " " + str(round(self.deviation, 2)),
                    (self.position[0]-100, self.position[1]-20), 1, 2, (0, 0, 255), 2)
        cv2.line(image, (self.pointer_x, self.y-10), (self.pointer_x, self.y+10), color, 3)
        cv2.line(image, (self.x1, self.y), (self.x2, self.y), color, 3)
    
    def track(self, mask):
        center_of_lane_x = None
        lane_x1 = None
        lane_x2 = None

        for i in range(self.width+1):
            if (np.any(mask[self.y, self.x1:self.position[0], 0] == 255) and \
                self.pointer_x <= self.position[0]) or \
                (np.any(mask[self.y, self.position[0]:self.x2, 0] == 255) and \
                self.pointer_x >= self.position[0]):
                self.is_active = True

            if self.is_active and mask[self.y, self.x1 + i, 0] == 255:
                lane_x1 = self.x1 + i
                lane_x2 = lane_x1
                i += 1
                while mask[self.y, self.x1 + i, 0] == 255 and i < self.width+1:
                    i += 1
                    lane_x2 += 1
                break

        if lane_x1 and lane_x2:
            center_of_lane_x = abs(lane_x2 + lane_x1)//2
            if center_of_lane_x < self.x1:
                center_of_lane_x = self.x1
            if center_of_lane_x > self.x2:
                center_of_lane_x = self.x2

        if center_of_lane_x:
            self.is_active = True
            self.pointer_x = center_of_lane_x

            self.deviation = (self.pointer_x-self.position[0])/(self.width//2)*100
        else:
            self.is_active = False

        return self.get_direction()

    def get_direction(self):
        if self.deviation < -20:
            return Direction.LEFT
        elif self.deviation > 20:
            return Direction.RIGHT
        return Direction.STRAIGHT

        


class LaneDetector:
    def __init__(self) -> None:
        self.distance_points = []
        self.center_pointer = [(1280//2, 720-30), (1280//2, 719)]
        self.distances = []

    def apply_mask(self, image):
        img = np.copy(image)
        hsv_transformed_frame = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        lower_red = np.array([0,50,50])
        upper_red = np.array([10,255,255])
        mask0 = cv2.inRange(hsv_transformed_frame, lower_red, upper_red)

        lower_red = np.array([150,50,50])
        upper_red = np.array([180,255,255])
        mask1 = cv2.inRange(hsv_transformed_frame, lower_red, upper_red)

        mask = mask0 + mask1
        mask = np.asarray(mask, np.uint8)

        blur = cv2.GaussianBlur(mask,(25,25),0)
        thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        return thresh
    
    def get_roi(self, image):
        new_image = np.zeros(image.shape)
        new_image[image.shape[0]//2:, :, :] = image[image.shape[0]//2:, :, :]
        return new_image
    
    def draw(self, image):
        cv2.line(image, self.center_pointer[0], self.center_pointer[1], (0, 0, 255), 3)

    def get_distance(self, mask, x=None):
        i = 0

        if not x:
            x = self.center_pointer[0][0]

        while i < self.center_pointer[1][1]+1:
            y = self.center_pointer[1][1]-i
            if mask[y, x, 0] == 255:
                return y
            i += 1
        return None

    def get_distances(self, mask):
        distance = self.get_distance(mask)
        if distance:
            self.distances.append(distance)
        else:
            self.distances = []

    def is_approaching_lane(self):
        if len(self.distances) > 3:
            if self.distances[0] < self.distances[-1]:
                return True
        return False
    
    def check_turn(self, mask):
        center_x = self.center_pointer[0][0]
        points = []
        for x in range(center_x - 50, center_x + 51, 10):
            distance = self.get_distance(mask, x)
            if distance is not None:
                points.append(distance)

        if len(points):
            popt, _ = curve_fit(lambda x, a, b: a * x + b, list(range(len(points))), points)
            a, _ = popt
            if a > 0:
                return Direction.LEFT
            else:
                return Direction.RIGHT
            
        return Direction.STRAIGHT
    
    def __call__(self, image):
        direction = None
        mask = self.apply_mask(image)
        mask = self.get_roi(mask)
        self.get_distances(mask)
        is_approaching_lane = self.is_approaching_lane()
        if is_approaching_lane:
            direction = self.check_turn(mask)
            cv2.putText(mask, direction.name + " turn detected", (100, 100), 1, 3, (0, 0, 255), 2)
        else:
            direction = Direction.STRAIGHT
            cv2.putText(mask, direction.name, (100, 100), 1, 3, (0, 0, 255), 2)

        self.draw(mask)

        return mask, direction
    

class Car:
    def __init__(self) -> None:
        self.lane_detector = LaneDetector()
        self.left_tracker = Tracker((300, 400), 190)
        self.right_tracker = Tracker((950, 400), 190)
        # self.left_tracker = Tracker((250, 500), 190)
        # self.right_tracker = Tracker((950, 500), 190)
        self.direction = Direction.STRAIGHT
        self.turn = None

    
    def __call__(self, frame):
        mask, turn = self.lane_detector(frame)

        self.turn = turn
        
        left_tracker_direction = self.left_tracker.track(mask)
        right_tracker_direction = self.right_tracker.track(mask)


        if self.turn.value == Direction.STRAIGHT.value:
            if left_tracker_direction.value == Direction.STRAIGHT.value:
                self.direction = left_tracker_direction
            elif right_tracker_direction.value == Direction.STRAIGHT.value:
                self.direction = right_tracker_direction
            else:
                self.direction = right_tracker_direction
        else:
            if self.turn.value == Direction.LEFT.value:
                if self.left_tracker.is_active:
                    self.direction = right_tracker_direction
                    self.turn = Direction.STRAIGHT
                else:
                    self.direction = self.turn
            elif self.turn.value == Direction.RIGHT.value:
                if self.right_tracker.is_active:
                    self.direction = left_tracker_direction
                    self.turn = Direction.STRAIGHT
                else:
                    self.direction = self.turn


        cv2.putText(mask, self.direction.name, (500 ,500), 1, 3, (0, 0, 255), 3)
        if self.turn.value != Direction.STRAIGHT.value:
            cv2.putText(mask, "Turn: " + self.turn.name, (500 ,550), 1, 3, (0, 0, 255), 3)


        self.left_tracker.draw(mask)
        self.right_tracker.draw(mask)

        return mask

def show_image():
    image = cv2.imread('image.png')
    mask = car(image)
    cv2.imshow('Frame', mask)
    cv2.waitKey()

def show_video():
    is_finish = False
    while not is_finish:
        is_finish = False
        cap = cv2.VideoCapture('przykladowa_trasa.mp4')
        # cap = cv2.VideoCapture('Untitled.mp4')
        success, image = cap.read()
        while success and not is_finish:
            success, image = cap.read()
            if not success:
                break
            
            
            mask = car(image)


            cv2.imshow('Frame', mask)

            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                is_finish = 1
            if key == ord('s'):
                if cv2.waitKey() & 0xFF == ord('q'):
                    is_finish = 1

        time.sleep(0.1) 


if __name__ == '__main__':
    car = Car()
    show_video()


