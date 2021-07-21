import threading
from pathlib import Path
import math
import cv2
import depthai
import numpy as np
from imutils.video import FPS
import datetime
import time
import requests
import telegram_send
import Jetson.GPIO as GPIO

nnPathPeople = str((Path(__file__).parent / Path('models/people.blob')).resolve().absolute()) #544x320 NN
nnPathMask = str((Path(__file__).parent / Path('models/facemask.blob')).resolve().absolute()) #300x300 NN
labelMapMask = ["background", "sin barbijo", "con barbijo"]

#variable de tiempo actual
Tiempo_AUX = datetime.datetime.now().timestamp()

#ubidots
TOKEN = "BBFF-8oWJDDDYhli48hqJlqpXarJOf2bxyy"
DEVICE_LABEL = "Monitoreo"

#-----------------configuracion de Jetson
GPIO.setmode(GPIO.BOARD)
LED_R=40
LED_G=38
LED_B=36
Buzzer=33
GPIO.setup(LED_R,GPIO.OUT)
GPIO.setup(LED_G,GPIO.OUT)
GPIO.setup(LED_B,GPIO.OUT)
GPIO.setup(Buzzer,GPIO.OUT)
delayT=0.2
#DISPLAY
toDisplay="2" # numbers and digits to display
delay = 0.005 # delay between digits refresh

dig1=23
dig2=21
selDigit = [dig1,dig2]
display_list = [24,22,12,15,13,19,16] 
digitDP = 11

GPIO.setwarnings(False)
for pin in display_list:
  GPIO.setup(pin,GPIO.OUT)
for pin in selDigit:
  GPIO.setup(pin,GPIO.OUT)
GPIO.setup(digitDP,GPIO.OUT)
GPIO.setwarnings(True)

arrSeg = [[0,0,0,0,0,0,1],\
          [1,0,0,1,1,1,1],\
          [0,0,1,0,0,1,0],\
          [0,0,0,0,1,1,0],\
          [1,0,0,1,1,0,0],\
          [0,1,0,0,1,0,0],\
          [0,1,0,0,0,0,0],\
          [0,0,0,1,1,1,1],\
          [0,0,0,0,0,0,0],\
          [0,0,0,0,1,0,0]]

GPIO.output(digitDP,0) # DOT pin

def showDisplay(digit):
 for i in range(0, 1): #loop on 4 digits selectors (from 0 to 3 included)
  sel = [1,1]
  sel[i] = 0
  GPIO.output(selDigit, sel) # activates selected digit
  if digit[i].replace(".", "") == " ": # space disables digit
   GPIO.output(display_list,0)
   continue
  numDisplay = int(digit[i].replace(".", ""))
  GPIO.output(display_list, arrSeg[numDisplay]) # segments are activated according to digit mapping
  if digit[i].count(".") == 1:
   GPIO.output(digitDP,1)
  else:
   GPIO.output(digitDP,0)
  time.sleep(delay)

def splitToDisplay (toDisplay): # splits string to digits to display
 arrToDisplay=list(toDisplay)
 for i in range(len(arrToDisplay)):
  if arrToDisplay[i] == ".": arrToDisplay[(i-1)] = arrToDisplay[(i-1)] + arrToDisplay[i] # dots are concatenated to previous array element
 while "." in arrToDisplay: arrToDisplay.remove(".") # array items containing dot char alone are removed
 return arrToDisplay

def post_request(payload):
    url = "http://industrial.api.ubidots.com"
    url = "{}/api/v1.6/devices/{}".format(url, DEVICE_LABEL)
    headers = {"X-Auth-Token": TOKEN, "Content-Type": "application/json"}

    status = 400
    attempts = 0
    while status >= 400 and attempts <= 5:
        req = requests.post(url=url, headers=headers, json=payload)
        status = req.status_code
        attempts += 1
        time.sleep(1)

    print(req.status_code, req.json())
    if status >= 400:
        print("[ERROR] Could not send data after 5 attempts, please check \
            your token credentials and internet connection")
        return False

    print("[INFO] request made properly, your device is updated")
    return True

def Funcion_tiempo():
    Tiempo_Actual = datetime.datetime.now().timestamp()
    Tiempo_Actual = Tiempo_Actual - Tiempo_AUX
    return Tiempo_Actual

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

def Wells_Riley(i, M):
    q = 120
    p = 0.52
    Q = 10    
    Exponente = -(i*q*p/10)/(360*Q*(1+M))   
    return Exponente

def Bayesian(i, M):
    P_CnB = 0.0212
    P_CnNB = 0.0706 
    Probabilidad = i*(M*P_CnB+(1-M)*P_CnNB)/60
    return Probabilidad

def P_individual_func(Pi, Pdist): 
    P_C = 0.212      
    Probabilidad = Pi +  Pdist
    return Probabilidad

def create_pipeline():
    print("Creating pipeline...")
    pipeline = depthai.Pipeline()

    print("Creating Color Camera...")
    cam = pipeline.createColorCamera()
    cam.setPreviewSize(544, 320)
    cam.setResolution(depthai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setInterleaved(False) #Set planar or interleaved data of preview output frames.
    cam.setBoardSocket(depthai.CameraBoardSocket.RGB)

    cam_xout = pipeline.createXLinkOut()
    cam_xout.setStreamName("cam_out")
    cam.preview.link(cam_xout.input)

    # NeuralNetwork
    print("Creating People Detection Neural Network...")
    people_nn = pipeline.createNeuralNetwork() #MobileNetDetectionNetwork
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
    monoLeft.setResolution(depthai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoLeft.setBoardSocket(depthai.CameraBoardSocket.LEFT)

    monoRight = pipeline.createMonoCamera()
    monoRight.setResolution(depthai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoRight.setBoardSocket(depthai.CameraBoardSocket.RIGHT)

    stereo = pipeline.createStereoDepth()
    spatialLocationCalculator = pipeline.createSpatialLocationCalculator()

    xoutDepth = pipeline.createXLinkOut()
    xoutDepth.setStreamName("depth")

    xoutSpatialData = pipeline.createXLinkOut()
    xoutSpatialData.setStreamName("spatialData")

    xinSpatialCalcConfig = pipeline.createXLinkIn()
    xinSpatialCalcConfig.setStreamName("spatialCalcConfig")

    # StereoDepth
    stereo.setOutputDepth(True) #Enable ‘depth’ stream
    stereo.setOutputRectified(False) #Optimizes computation on device side when disabled.
    stereo.setConfidenceThreshold(255)
    stereo.setLeftRightCheck(False) #Computes and combines disparities in both L-R and R-L directions, and combine them
    stereo.setSubpixel(False) #Calcula la disparidad con interpolación de subpíxeles

    monoLeft.out.link(stereo.left)
    monoRight.out.link(stereo.right)

    spatialLocationCalculator.passthroughDepth.link(xoutDepth.input) #entrada de datos
    stereo.depth.link(spatialLocationCalculator.inputDepth)

    topLeft = depthai.Point2f(0.4, 0.4)
    bottomRight = depthai.Point2f(0.6, 0.6)

    spatialLocationCalculator.setWaitForConfigInput(False)
    config = depthai.SpatialLocationCalculatorConfigData()
    config.depthThresholds.lowerThreshold = 100 #umbral inferior en milímetros para los valores de profundidad
    config.depthThresholds.upperThreshold = 10000
    config.roi = depthai.Rect(topLeft, bottomRight) #ROI region de interes
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
        self.cam_out = self.device.getOutputQueue(name="cam_out", maxSize=1, blocking=False)
        self.Top_view_Image = np.zeros((320,544,3), dtype=np.uint8)
        self.frame = None

        self.bboxes = []
        self.depth_dist = []
        self.min_dists = []
        self.P_dists = []
        self.mask_bboxes = []
        self.mask_detections = []
        self.current_bbox = None
        self.Pi = 0
        self.toDisplay = "  " 
        self.Tiempo_segundos = 0
        self.Compliance_People = 0
        self.Compliance_mask = 0
        self.Npersonas = 0

        self.running = True
        self.fps = FPS()
        self.fps.start()

    def people_thread(self):
        print("people thread")        
        people_nn = self.device.getOutputQueue(name="people_nn", maxSize=1, blocking=False)
        mask_in = self.device.getInputQueue("mask_in") #frames de entrada para mask_nn
        while self.running:
            if self.frame is None:
                continue
            try:
                bboxes = np.array(people_nn.get().getFirstLayerFp16())
            except RuntimeError as ex:
                continue
            bboxes = bboxes.reshape((bboxes.size // 7, 7)) #image_id, label, confidence, x_min, y_min, x_max, y_max
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
            P_dists = list()
            for p1 in z_dists:
                min_dist = math.inf                
                P_dist = 0
                counter = 0                
                for p2 in z_dists:
                    if p1 != p2:
                        dist = calculate_distance(p1,p2)
                        x=dist/1000
                        if x > 1 and x <10:
                            P_dist = -0.2889*x + 2.8889
                        elif  x >= 0 and x <= 1:
                            P_dist = -10.2*x + 12.8
                        else:
                            P_dist = 0
                        min_dist = min(dist, min_dist)                        
                        counter += 1
                        print(P_dist)
                
                min_dits.append(min_dist)
                P_dists.append(P_dist)

            self.depth_dist = z_dists
            self.min_dists = min_dits
            self.P_dists = P_dists

    def alarm_thread(self):
        print("Alarm thread")
        while self.running:
            if(self.Pi>10):
                i=0
                while i<3:
                    GPIO.output(LED_R,1)
                    GPIO.output(Buzzer,1)
                    time.sleep(delayT)

                    GPIO.output(LED_R,0)  
                    GPIO.output(Buzzer,0)
                    time.sleep(delayT)
                    i+=1
                time.sleep(delayT*3)
    
    def Display_thread(self):
        print("Display thread")
        try:
            while self.running:
                showDisplay(splitToDisplay(self.toDisplay))
        except:
            print('An exception occurred')
            GPIO.cleanup()

    def monitoreo_thread(self):
        print("Monitoreo thread")
        i=0
        while self.running:
            payload = {"cumplimiento-de-distancia": self.Compliance_People,
                        "cumplimiento-del-barbijo": int(self.Compliance_mask),
                        "numero-de-personas": self.Npersonas,
                        "probabilidad-de-contagio": round(self.Pi, 3)}

            if int(self.Tiempo_segundos)%30 ==0 and int(self.Tiempo_segundos) != 0 or i==0:
                i=1
                print("[INFO] Attemping to send data")
                #post_request(payload)
                print("[INFO] finished")
                if self.Pi > 10:
                    telegram_send.send(messages=["ALERTA: Riesgo de propagación " + str(self.Pi) + "%"])

    def get_frame(self, retries=0):
        return np.array(self.cam_out.get().getData()).reshape((3, 320, 544)).transpose(1, 2, 0).astype(np.uint8)

    def run(self):
        depthQueue = device.getOutputQueue(name="depth", maxSize=1, blocking=False)
        self.threads = [
            threading.Thread(target=self.people_thread),
            threading.Thread(target=self.mask_thread),
            threading.Thread(target=self.depth_thread),
            threading.Thread(target=self.alarm_thread),
            threading.Thread(target=self.Display_thread),
            threading.Thread(target=self.monitoreo_thread)
        ]
        for thread in self.threads:
            thread.start()
        
        contador_segundos = 0
        aux_100 = 0
        contador_calculos = 0
        E=0
        while True:
            try:
                new_frame = self.get_frame()
                inDepth = depthQueue.get()
            except RuntimeError:
                continue

            self.fps.update()
            self.frame = new_frame
            self.debug_frame = self.frame.copy()
            self.Compliance_People = 0
            self.Compliance_mask = 0
            #--------------------------------------------------------------MASK
            for raw_bbox, label in zip(self.mask_bboxes, self.mask_detections):
                bbox = frame_norm(self.frame, raw_bbox)
                if str(labelMapMask[int(label)]) == 'con barbijo':
                    color = (0, 255, 0)
                    self.Compliance_mask += 1
                else:
                    color = (0, 0, 255)
                cv2.rectangle(self.debug_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

            #-------------------------------------------------------------------------------------------------PEOPLE
            for raw_bbox, depth, min_dist, P_dist in zip(self.bboxes, self.depth_dist, self.min_dists, self.P_dists):
                bbox = frame_norm(self.frame, raw_bbox)               
                if min_dist == math.inf or int(min_dist)/1000 > 2:                    
                    color = (0, 255, 0)
                    self.Compliance_People =+ 1
                else:                    
                    color = (0, 0, 255)
                cv2.circle(self.Top_view_Image, (int(depth[0]),int(depth[2])), 50, (255, 255, 255), -1) 
                if min_dist != math.inf:
                    x = int(min_dist)/1000
                    Pindividual = P_individual_func(self.Pi, P_dist)
                    cv2.putText(self.debug_frame, f"Dmin: {str(round(x,1))} m", (bbox[0] + 10, bbox[1] + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 0))
                    cv2.putText(self.debug_frame, f"Pin: {str(round(Pindividual,2))} %", (bbox[0] + 10, bbox[1] + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 0))

                cv2.rectangle(self.debug_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
#---------------------------------------------------------------------------------------------------------------Calculos          
         
            self.Tiempo_segundos=int(Funcion_tiempo())
            self.Npersonas = int(len(self.bboxes))
            if len(self.bboxes):
                self.Compliance_mask = (self.Compliance_mask / self.Npersonas) * 100
                if len(self.bboxes)>1: self.Compliance_People = (self.Compliance_People / self.Npersonas) * 100 
                else: self.Compliance_People = 1

            if self.Tiempo_segundos != contador_segundos and self.Tiempo_segundos > contador_segundos:
                E += Wells_Riley(self.Npersonas, self.Compliance_mask/100)                
                self.Pi = (1-(2.71828 ** E))*100
                contador_segundos = self.Tiempo_segundos
                self.toDisplay=str(int(self.Pi))            
                if self.Pi != contador_calculos:
                    aux_100 += 1
                    contador_calculos = self.Pi
                    print (aux_100, "Pi:", round(self.Pi, 2), "N:", self.Npersonas, "Mask:", self.Compliance_mask, "Dist:",self.Compliance_People*100)

#----cv.imshow
            color = (255, 255, 255)
            Data_image = np.zeros((320,100,3), dtype=np.uint8)            
            cv2.putText(Data_image, f"Pi: {round(self.Pi, 6)} %", (2, Data_image.shape[0] - 15),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.4, color)
            cv2.putText(Data_image, f"Barbijo: {round(self.Compliance_mask,2)} %", (2, Data_image.shape[0] - 45),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.4, color)
            cv2.putText(Data_image, f"Personas: {self.Npersonas} ", (2, Data_image.shape[0] - 60),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.4, color)
            cv2.putText(Data_image, f"tiempo: {int(self.Tiempo_segundos)} s", (2, Data_image.shape[0] - 315),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.4, color)
            cv2.putText(Data_image, f"DS: {round(self.Compliance_People*100,2)} %", (2, Data_image.shape[0] - 30),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.4, color)
            Out_frames = np.concatenate((Data_image, self.debug_frame), axis=1)
            Out_frames = np.concatenate((Out_frames, self.Top_view_Image), axis=1)
            #cv2.imshow("Camera view", self.debug_frame)
            cv2.imshow("datos", Out_frames)
            
            if cv2.waitKey(1) == ord('q'):
                cv2.destroyAllWindows()
                break

        self.fps.stop()
        print("FPS: {:.2f}".format(self.fps.fps()))
        cv2.destroyAllWindows()
        for i in range(1, 5):
            cv2.waitKey(1)
        self.running = False

dev, conf = create_pipeline()
with depthai.Device(dev) as device:
    app = Main(device, conf)
    app.run()

for thread in app.threads:
    thread.join()