import threading
from pathlib import Path
import math
import cv2
import depthai
import numpy as np
from imutils.video import FPS

nnPathPeople = str((Path(__file__).parent / Path('models/people.blob')).resolve().absolute()) #544x320 NN
nnPathMask = str((Path(__file__).parent / Path('models/facemask.blob')).resolve().absolute()) #300x300 NN
labelMapMask = ["background", "sin barbijo", "con barbijo"]
stepSize = 0.05

def frame_norm(frame, bbox):
    norm_vals = np.full(len(bbox), frame.shape[0])
    norm_vals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * norm_vals).astype(int)

def to_planar(arr: np.ndarray, shape: tuple) -> list:
    return [val for channel in cv2.resize(arr, shape).transpose(2, 0, 1) for y_col in channel for val in y_col]

def calculate_distance(point1, point2):
    x1, y1, z1 = point1
    x2, y2, z2 = point2
    dx, dy, dz = x1 - x2, y1 - y2, z1 - z2
    distance = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    return distance

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

    #---------------------------------------spatial calculator
    # Define a source - two mono (grayscale) cameras
    monoLeft = pipeline.createMonoCamera()
    monoRight = pipeline.createMonoCamera()
    stereo = pipeline.createStereoDepth()
    spatialLocationCalculator = pipeline.createSpatialLocationCalculator()

    xoutDepth = pipeline.createXLinkOut()
    xoutSpatialData = pipeline.createXLinkOut()
    xinSpatialCalcConfig = pipeline.createXLinkIn()

    xoutDepth.setStreamName("depth")
    xoutSpatialData.setStreamName("spatialData")
    xinSpatialCalcConfig.setStreamName("spatialCalcConfig")

    # MonoCamera
    monoLeft.setResolution(depthai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoLeft.setBoardSocket(depthai.CameraBoardSocket.LEFT)
    monoRight.setResolution(depthai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoRight.setBoardSocket(depthai.CameraBoardSocket.RIGHT)

    outputDepth = True
    outputRectified = False
    lrcheck = False
    subpixel = False

    # StereoDepth
    stereo.setOutputDepth(outputDepth)
    stereo.setOutputRectified(outputRectified)
    stereo.setConfidenceThreshold(255)

    stereo.setLeftRightCheck(lrcheck)
    stereo.setSubpixel(subpixel)

    monoLeft.out.link(stereo.left)
    monoRight.out.link(stereo.right)

    spatialLocationCalculator.passthroughDepth.link(xoutDepth.input)
    stereo.depth.link(spatialLocationCalculator.inputDepth)

    topLeft = depthai.Point2f(0.4, 0.4)
    bottomRight = depthai.Point2f(0.6, 0.6)

    spatialLocationCalculator.setWaitForConfigInput(False)
    config = depthai.SpatialLocationCalculatorConfigData()
    config.depthThresholds.lowerThreshold = 100
    config.depthThresholds.upperThreshold = 10000
    config.roi = depthai.Rect(topLeft, bottomRight)
    spatialLocationCalculator.initialConfig.addROI(config)
    spatialLocationCalculator.out.link(xoutSpatialData.input)
    xinSpatialCalcConfig.out.link(spatialLocationCalculator.inputConfig)

    # Pipeline defined, now the device is assigned and pipeline is started
    device = depthai.Device(pipeline)
    device.startPipeline()
    return pipeline, config

class Main:
    def __init__(self, device, config):
        self.device = device
        self.config = config
        print("Starting pipeline...")
        self.device.startPipeline()
        self.cam_out = self.device.getOutputQueue("cam_out")
        self.frame = None

        self.bboxes = []
        self.depth_dist = []
        self.min_dists = []
        self.mask_bboxes = []
        self.mask_detections = []
        self.current_bbox = None

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

            cfg = depthai.SpatialLocationCalculatorConfig()
            for bbox in self.bboxes:
                bbox = frame_norm(self.frame, bbox)
                topLeft = depthai.Point2f(bbox[0], bbox[1])
                bottomRight = depthai.Point2f(bbox[2], bbox[3])
                self.config.roi = depthai.Rect(topLeft, bottomRight)
                cfg.addROI(self.config)

            if len(self.bboxes) > 0:
                self.device.getInputQueue("spatialCalcConfig").send(cfg)

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

    def depth_thread(self):
        print("depth thread")
        # Output queue will be used to get the depth frames from the outputs defined above
        spatialCalcQueue = self.device.getOutputQueue(name="spatialData", maxSize=1, blocking=False)

        while self.running:
            try:
                inDepthAvg = spatialCalcQueue.get()  # blocking call, will wait until a new data has arrived
            except RuntimeError as ex:
                continue
            spatialData = inDepthAvg.getSpatialLocations()
            z_dists = list()

            for depthData in spatialData:
                z_dists.append([depthData.spatialCoordinates.x, depthData.spatialCoordinates.y, depthData.spatialCoordinates.z])

            min_dits = list()
            for p1 in z_dists:
                min_dist = math.inf
                for p2 in z_dists:
                    if p1 != p2:
                        dist = calculate_distance(p1,p2)
                        min_dist = min(dist, min_dist)
                min_dits.append(min_dist)

            self.depth_dist = z_dists
            self.min_dists = min_dits


    def get_frame(self, retries=0):
        return np.array(self.cam_out.get().getData()).reshape((3, 320, 544)).transpose(1, 2, 0).astype(np.uint8)

    def run(self):
        depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
        self.threads = [
            threading.Thread(target=self.people_thread),
            threading.Thread(target=self.mask_thread),
            threading.Thread(target=self.depth_thread)
        ]
        for thread in self.threads:
            thread.start()

        while True:
            try:
                new_frame = self.get_frame()
                inDepth = depthQueue.get()
            except RuntimeError:
                continue

            self.fps.update()
            self.frame = new_frame
            self.debug_frame = self.frame.copy()

            for raw_bbox, depth, min_dist in zip(self.bboxes, self.depth_dist, self.min_dists):
                bbox = frame_norm(self.frame, raw_bbox)
                cv2.rectangle(self.debug_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (10, 245, 10), 2)
                cv2.putText(self.debug_frame, "x: "+str(round(depth[0],2)), (bbox[0] + 10, bbox[1] + 20),
                            cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 10))
                cv2.putText(self.debug_frame, "y: "+str(round(depth[1],2)), (bbox[0] + 10, bbox[1] + 35),
                            cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 10))
                cv2.putText(self.debug_frame, "z: "+str(round(depth[2],2)), (bbox[0] + 10, bbox[1] + 50),
                            cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 10))
                cv2.putText(self.debug_frame, "D: "+str(round(min_dist,2)), (bbox[0] + 10, bbox[1] + 65),
                            cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 10))

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
dev, conf = create_pipeline()
with depthai.Device(dev) as device:
    app = Main(device, conf)
    app.run()

for thread in app.threads:
    thread.join()
