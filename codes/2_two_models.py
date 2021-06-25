import argparse
import queue
import threading
import signal
from pathlib import Path

import cv2
import depthai
import numpy as np
from imutils.video import FPS
from math import cos, sin

nnPathPeople = str((Path(__file__).parent / Path('models/people.blob')).resolve().absolute()) #544x320 NN
nnPathMask = str((Path(__file__).parent / Path('models/facemask.blob')).resolve().absolute()) #300x300 NN

labelMapMask = ["background", "sin barbijo", "con barbijo"]
print("---------------------CORRIENDO: Distance-2.py----------------------")
def frame_norm(frame, bbox):
    norm_vals = np.full(len(bbox), frame.shape[0])
    norm_vals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * norm_vals).astype(int)


def to_planar(arr: np.ndarray, shape: tuple) -> list:
    return [val for channel in cv2.resize(arr, shape).transpose(2, 0, 1) for y_col in channel for val in y_col]

def create_pipeline():
    print("Creating pipeline...")
    pipeline = depthai.Pipeline()

    print("Creating Color Camera...")
    cam = pipeline.createColorCamera()
    cam.setPreviewSize(544, 320)
    cam.setResolution(depthai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setInterleaved(False)
    cam.setBoardSocket(depthai.CameraBoardSocket.RGB)

    cam_xout = pipeline.createXLinkOut()
    cam_xout.setStreamName("cam_out")
    cam.preview.link(cam_xout.input)

    # NeuralNetwork
    print("Creating People Detection Neural Network...")
    people_nn = pipeline.createNeuralNetwork()
    people_nn.setBlobPath(nnPathPeople)

    cam.preview.link(people_nn.input)

    people_nn_xout = pipeline.createXLinkOut()
    people_nn_xout.setStreamName("people_nn")
    people_nn.out.link(people_nn_xout.input)

    # NeuralNetwork
    print("Creating Mask Detection Neural Network...")
    mask_nn = pipeline.createNeuralNetwork()
    mask_nn.setBlobPath(nnPathMask)

    mask_nn_xin = pipeline.createXLinkIn()
    mask_nn_xin.setStreamName("mask_in")
    mask_nn_xin.out.link(mask_nn.input)
    land_nn_xout = pipeline.createXLinkOut()
    land_nn_xout.setStreamName("mask_nn")
    mask_nn.out.link(land_nn_xout.input)

    return pipeline

class Main:
    def __init__(self, device):
        self.device = device
        print("Starting pipeline...")
        self.device.startPipeline()
        self.cam_out = self.device.getOutputQueue("cam_out")
        self.frame = None

        self.bboxes = []
        self.mask_bboxes = []
        self.mask_detections = []

        self.running = True
        self.fps = FPS()
        self.fps.start()

    def people_thread(self):
        print("people thread")
        people_nn = self.device.getOutputQueue("people_nn")
        mask_in = self.device.getInputQueue("mask_in")

        while self.running:
            if self.frame is None:
                continue
            try:
                bboxes = np.array(people_nn.get().getFirstLayerFp16())
            except RuntimeError as ex:
                continue
            bboxes = bboxes.reshape((bboxes.size // 7, 7))
            self.bboxes = bboxes[bboxes[:, 2] > 0.7][:, 3:7] #fancy indexing

            maks_data = depthai.NNData()
            maks_data.setLayer("0", to_planar(self.frame, (300, 300)))
            mask_in.send(maks_data)

    def mask_thread(self):
        print("mask thread")

        mask_nn = self.device.getOutputQueue(name="mask_nn", maxSize=1, blocking=False)

        while self.running:
            try:
                bboxes = np.array(mask_nn.get().getFirstLayerFp16())
            except RuntimeError as ex:
                continue
            bboxes = bboxes.reshape((bboxes.size // 7, 7))
            self.mask_bboxes = bboxes[bboxes[:, 2] > 0.7][:, 3:7]
            self.mask_detections = bboxes[bboxes[:, 2] > 0.7][:, 1]

    def get_frame(self, retries=0):
        return np.array(self.cam_out.get().getData()).reshape((3, 320, 544)).transpose(1, 2, 0).astype(np.uint8)

    def run(self):
        self.threads = [
            threading.Thread(target=self.people_thread),
            threading.Thread(target=self.mask_thread),
        ]
        for thread in self.threads:
            thread.start()

        while True:
            try:
                new_frame = self.get_frame()
            except RuntimeError:
                continue

            self.fps.update()
            self.frame = new_frame
            self.debug_frame = self.frame.copy()

            for raw_bbox in self.bboxes:
                bbox = frame_norm(self.frame, raw_bbox)
                cv2.rectangle(self.debug_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (10, 245, 10), 2)

            for raw_bbox, label in zip(self.mask_bboxes, self.mask_detections):
                bbox = frame_norm(self.frame, raw_bbox)
                cv2.rectangle(self.debug_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (10, 215, 10), 2)
                cv2.putText(self.debug_frame, str(labelMapMask[int(label)]), (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (10, 215, 10))

            cv2.imshow("Camera view", self.debug_frame)

            if cv2.waitKey(1) == ord('q'):
                cv2.destroyAllWindows()
                break

        self.fps.stop()
        print("FPS: {:.2f}".format(self.fps.fps()))
        cv2.destroyAllWindows()
        for i in range(1, 5):  # https://stackoverflow.com/a/25794701/5494277
            cv2.waitKey(1)
        self.running = False

with depthai.Device(create_pipeline()) as device:
    app = Main(device)
    app.run()

for thread in app.threads:
    thread.join()
