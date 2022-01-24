import threading
from pathlib import Path
import math
import cv2
import depthai
import numpy as np
from imutils.video import FPS
import datetime
import tkinter
from tkinter import ttk
import sys
import pickle as pk
import requests
import telegram_send
import time

nnPathPeople = str((Path(__file__).parent / Path('models/people.blob')).resolve().absolute()) #544x320 NN
nnPathMask = str((Path(__file__).parent / Path('models/facemask.blob')).resolve().absolute()) #300x300 NN
labelMapMask = ["background", "sin barbijo", "con barbijo"]
#ubidots
TOKEN = "BBFF-8oWJDDDYhli48hqJlqpXarJOf2bxyy"
DEVICE_LABEL = "Monitoreo"
#variable de tiempo actual
Tiempo_AUX = datetime.datetime.now().timestamp()
#ventana inicial
Color_Background='dark slate gray'
Font_Color='snow'
ventana = tkinter.Tk()
ventana.configure(background=Color_Background)
ventana.title("Initial Setup")
#------Global Variables-----
wells_q = 0.1
wells_p = 0.1
wells_Q = 0.1
Prob_PC = 100
calc_time = 60
Dim_X = 0.1
Dim_Z = 0.1
Modo_variable = ""
Init_Configuration=True

# Save data file
PIK = "Input_data.pk"
PIK_out = "Output_data.pk"
pik_gauss = "Gauss.pk"

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

def roomdim():
    global Dim_X
    global Dim_Z
    return Dim_X, Dim_Z

def ConsideracionDeDistancias(x_personM,z_personM):
    xroom, zroom = roomdim()
    xroom = int(xroom)
    zroom = int(zroom)
    xpix = 664
    ypix = xpix * xroom / zroom
    ypix = int(ypix)
    x_person = int(((x_personM + int(Dim_X) / 2) / int(Dim_X)) * ypix)
    z_person = int((z_personM / int(Dim_Z)) * xpix)
    return x_person, z_person

def Wells_Riley(i, M):
    global wells_q
    global wells_p
    global wells_Q
    global Prob_PC
    global Modo_variable
    deposicion = 6
    if(Modo_variable=="Debug"):
        time_wells = 60
    else:
        time_wells = 3600

    Exponente = -(i*(float(Prob_PC)/100)*float(wells_q)*float(wells_p))/(deposicion*time_wells*float(wells_Q)*(1+M))
    return Exponente

def Bayesian(i, M):
    P_CnB = 0.0212
    P_CnNB = 0.0706 
    Probabilidad = i*(M*P_CnB+(1-M)*P_CnNB)/60
    return Probabilidad

def P_individual_func(Pi, Pdist): 
    P_C = 0.212
    #Probabilidad = Pi + P_C*Pdist  # tomando en cuenta la probabilidad de infectados
    Probabilidad = Pi + Pdist #tomando en cuenta a todos como infectados
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
        xroom, zroom = roomdim()
        xroom = int(xroom)
        zroom = int(zroom)
        xpix = 664
        ypix = int(xpix * xroom / zroom)
        self.Top_view_Image = np.zeros((ypix, xpix, 3), dtype=np.uint8)
        self.Heatmap_View = np.zeros((ypix, xpix, 3), dtype=np.uint8)
        self.Bool_telegram = False
        self.Time_telegram = self.Tiempo_segundos

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
                        #print(P_dist)
                
                min_dits.append(min_dist)
                P_dists.append(P_dist)

            self.depth_dist = z_dists
            self.min_dists = min_dits
            self.P_dists = P_dists

    def monitoreo_thread(self):
        print("Monitoreo thread")
        i=0
        while self.running:
            payload = {"cumplimiento-de-distancia": self.Compliance_People,
                        "cumplimiento-del-barbijo": int(self.Compliance_mask),
                        "numero-de-personas": self.Npersonas,
                        "probabilidad-de-contagio": round(self.Pi, 3)}
            if (self.Pi > 10):  # asd umbral para activar la alerta y los mensajes de alerta
                if (self.Bool_telegram == False):
                    self.Time_telegram = self.Tiempo_segundos
                    print('Tiempo de alarma: ', self.Time_telegram)
                    self.Bool_telegram = True
                    telegram_send.send(messages=["ALERTA: Riesgo de propagación " + str(self.Pi) + "%"])

                if int(self.Tiempo_segundos - self.Time_telegram) % 60 == 0 and self.Bool_telegram == True:
                    print('Sending Telegram...')
                    telegram_send.send(messages=["ALERTA: Riesgo de propagación " + str(self.Pi) + "%"])

            if int(self.Tiempo_segundos)%60 ==0 and int(self.Tiempo_segundos) != 0 or i==0: #asd Tiempo de actualizacion Ubidots
                i=1
                print("[INFO] Attemping to send data")
                post_request(payload)
                print("[INFO] finished")

    def get_frame(self, retries=0):
        return np.array(self.cam_out.get().getData()).reshape((3, 320, 544)).transpose(1, 2, 0).astype(np.uint8)

    def run(self):
        f_gauss = open(pik_gauss, "wb")
        depthQueue = self.device.getOutputQueue(name="depth", maxSize=1, blocking=False)
        self.threads = [
            threading.Thread(target=self.people_thread),
            threading.Thread(target=self.mask_thread),
            threading.Thread(target=self.depth_thread),
            threading.Thread(target=self.monitoreo_thread)
        ]
        for thread in self.threads:
            thread.start()
        
        contador_segundos = 0
        aux_100 = 0
        contador_calculos = 0
        E=0

        xroom, zroom = roomdim()
        xroom = int(xroom)
        zroom = int(zroom)
        xpix = 664
        ypix = xpix * xroom / zroom
        ypix = int(ypix)
        anglegrad = (180 - 68.8) / 2
        angle = anglegrad * (math.pi/180)
        Xpixeles = math.tan(angle) * (ypix / 2)
        Xpixeles = int (Xpixeles)
        #print('Ypix, Angulo y xpixeles ', ypix, anglegrad, Xpixeles)
        while True:
            try:
                new_frame = self.get_frame()
                inDepth = depthQueue.get()
            except RuntimeError:
                continue

            self.fps.update()
            self.frame = new_frame
            self.debug_frame = self.frame.copy()
            self.debug_TopView = self.Heatmap_View.copy()
            self.Compliance_People = 0
            self.Compliance_mask = 0
            self.Tiempo_segundos = int(Funcion_tiempo())
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
                if min_dist == math.inf or int(min_dist) / 1000 > 2:
                    color = (0, 255, 0)
                    self.Compliance_People = + 1
                else:
                    color = (0, 0, 255)

                if min_dist != math.inf:
                    x = int(min_dist) / 1000
                    Pindividual = P_individual_func(self.Pi, P_dist)
                    cv2.putText(self.debug_frame, f"Dmin: {str(round(x, 1))} m", (bbox[0] + 10, bbox[1] + 65),
                                cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 0))
                    cv2.putText(self.debug_frame, f"Pin: {str(round(Pindividual, 2))} %", (bbox[0] + 10, bbox[1] + 80),
                                cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 0))

                cv2.rectangle(self.debug_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                cv2.putText(self.debug_frame, "x: " + str(round(depth[0], 2)), (bbox[0] + 10, bbox[1] + 20),
                            cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 10))
                cv2.putText(self.debug_frame, "z: " + str(round(depth[2], 2)), (bbox[0] + 10, bbox[1] + 50),
                            cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 10))
                #distancia v2
                x_person, z_person = ConsideracionDeDistancias(depth[0],depth[2])
                cv2.circle(self.debug_TopView, (z_person, x_person), 10, (0, 0, 255), -1)

                pk.dump(int(z_person), f_gauss)
                pk.dump(int(x_person), f_gauss)
                print('Gauss X: ',int(x_person),'Gauss Z: ',int(z_person))
                cv2.circle(self.Heatmap_View, (z_person, x_person), 10, (0, 150, 150), -1)
            
#---------------------------------------------------------------------------------------------------------------Calculos          
         

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
            color = (0, 255, 255)
            Data_image = np.zeros((320, 120, 3), dtype=np.uint8)
            cv2.putText(Data_image, f"Pi: {round(self.Pi, 6)} %", (2, Data_image.shape[0] - 15),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.4, color)
            cv2.putText(Data_image, f"Barbijo: {round(self.Compliance_mask, 2)} %", (2, Data_image.shape[0] - 45),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.4, color)
            cv2.putText(Data_image, f"Personas: {self.Npersonas} ", (2, Data_image.shape[0] - 60),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.4, color)
            cv2.putText(Data_image, f"tiempo: {int(self.Tiempo_segundos)} s", (2, Data_image.shape[0] - 310),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.4, color)
            cv2.putText(Data_image, f"t. exp: {int(aux_100)} s", (2, Data_image.shape[0] - 295),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.4, color)
            cv2.putText(Data_image, f"DS: {round(self.Compliance_People, 2)} %", (2, Data_image.shape[0] - 30),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.4, color)

            # ----------------------------------------------------------------------------------------Field of view lines
            # cv2.imshow("Propagation", self.Heatmap_View)

            cv2.line(self.debug_TopView, (30, int(ypix/2-15)), (int(Xpixeles), 0), (255, 255, 255), 3)
            cv2.line(self.debug_TopView, (30, int(ypix/2+15)), (int(Xpixeles), ypix), (255, 255, 255), 3)
            cv2.rectangle(self.debug_TopView, (0, int(ypix/2-15)), (30, int(ypix/2+15)), (255, 255, 255), 5)
            cv2.putText(self.debug_TopView, "C", (5, int(ypix/2+10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

            Out_frames = np.concatenate((Data_image, self.debug_frame), axis=1)
            # Top View Size: 390x664
            Out_frames = np.concatenate((Out_frames, self.debug_TopView), axis=0)
            cv2.imshow("Frame", Out_frames)

            if cv2.waitKey(1) == ord('q'):
                cv2.destroyAllWindows()
                break

        self.fps.stop()
        print("FPS: {:.2f}".format(self.fps.fps()))
        cv2.destroyAllWindows()
        for i in range(1, 5):
            cv2.waitKey(1)
        self.running = False

def Second_Code():
    dev, conf = create_pipeline()

    with depthai.Device(dev) as device:
        print('Data Device:', device)
        app = Main(device, conf)
        app.run()

    for thread in app.threads:
        thread.join()

def windows_events(event):
    global wells_q
    global wells_p
    global wells_Q
    global Prob_PC
    global calc_time
    global Dim_X
    global Dim_Z
    global Modo_variable
    global Init_Configuration

    if Combo_modo.get() == "Debug":
        calc_time = 60
    else:
        calc_time = 1

    wells_q = Entry_wells_q.get()
    wells_p = Entry_wells_p.get()
    wells_Q = Entry_wells_Q.get()
    Prob_PC = Entry_PC.get()
    Dim_X = Entry_DimX.get()
    Dim_Z = Entry_DimZ.get()
    Modo_variable = Combo_modo.get()
    print("MODO:",Combo_modo.get(), " EQ:", Combo_ecuacion.get(), " DimX:",Dim_X, " DimY:",Dim_Z, " PC:",Prob_PC, " q:",wells_q, " p:",wells_p, " Q:",wells_Q)
    if Combo_ecuacion.get()=="Wells-Riley":
        Label_wells_q.grid(row=6, column=1, columnspan=1)
        Label_wells_p.grid(row=7, column=1, columnspan=1)
        Label_wells_Q.grid(row=8, column=1, columnspan=1)
        Entry_wells_q.grid(row=6, column=2, columnspan=2)
        Entry_wells_p.grid(row=7, column=2, columnspan=2)
        Entry_wells_Q.grid(row=8, column=2, columnspan=2)
    else:
        Label_wells_q.grid_forget()
        Label_wells_p.grid_forget()
        Label_wells_Q.grid_forget()
        Entry_wells_q.grid_forget()
        Entry_wells_p.grid_forget()
        Entry_wells_Q.grid_forget()
    if Combo_ecuacion.get()=="Probability V1":
        cond_F.grid(row=6, column=1, columnspan=1)
        cond_M.grid(row=7, column=1, columnspan=1)
        Entry_cond_F.grid(row=6, column=2, columnspan=2)
        Entry_cond_M.grid(row=7, column=2, columnspan=2)
    else:
        cond_F.grid_forget()
        cond_M.grid_forget()
        Entry_cond_F.grid_forget()
        Entry_cond_M.grid_forget()
    if event == 'Iniciar':
        Init_Configuration = False
        ventana.destroy()
        Second_Code()
    elif event == 'Cerrar':
        sys.exit()

if(Init_Configuration==True):
    ##----------Labels
    Label_Modo = tkinter.Label(ventana, text = "Mode", width = 10, height = 2, fg= Font_Color, bg= Color_Background)
    Label_Dimension = tkinter.Label(ventana, text = "Room dimension (X,Z)", width = 30, height = 2, fg= Font_Color, bg= Color_Background)
    Label_Infectados = tkinter.Label(ventana, text = "Infected (%)", width = 10, height = 2, fg= Font_Color, bg= Color_Background)
    Label_Ecuacion = tkinter.Label(ventana, text = "Equation", width = 10, height = 2, fg= Font_Color, bg= Color_Background)
    Label_Parámetros = tkinter.Label(ventana, text = "Equation Factors", width = 40, height = 2, fg= 'snow', bg= 'chocolate2')
    Label_wells_q = tkinter.Label(ventana, text = "q", width = 10, height = 2, fg= Font_Color, bg= Color_Background)
    Label_wells_p = tkinter.Label(ventana, text = "p", width = 10, height = 2, fg= Font_Color, bg= Color_Background)
    Label_wells_Q = tkinter.Label(ventana, text = "Q", width = 10, height = 2, fg= Font_Color, bg= Color_Background)
    cond_F = tkinter.Label(ventana, text="F", width=10, height=2, fg=Font_Color, bg=Color_Background)
    cond_M = tkinter.Label(ventana, text="M", width=10, height=2, fg=Font_Color, bg=Color_Background)
    ##----------Options
    Options1 = [
        "Debug",
        "Monitoring"
    ]
    Combo_modo = ttk.Combobox(ventana, value=Options1)
    Combo_modo.current(0)
    Combo_modo.bind("<<ComboboxSelected>>",windows_events)

    Options2 = [
        "Wells-Riley",
        "Probability V1"
    ]
    Combo_ecuacion = ttk.Combobox(ventana, value=Options2)
    Combo_ecuacion.current(0)
    Combo_ecuacion.bind("<<ComboboxSelected>>", windows_events)

    ##----------Entrys
    Entry_DimX = tkinter.Entry(ventana)
    Entry_DimZ = tkinter.Entry(ventana)
    Entry_PC = tkinter.Entry(ventana)
    Entry_wells_q = tkinter.Entry(ventana)
    Entry_wells_p = tkinter.Entry(ventana)
    Entry_wells_Q = tkinter.Entry(ventana)
    Entry_cond_F = tkinter.Entry(ventana)
    Entry_cond_M = tkinter.Entry(ventana)
    ##----------Buttons
    Button_iniciar = tkinter.Button(ventana, text= "Start", command=lambda: windows_events("Iniciar"))
    Button_cerrar = tkinter.Button(ventana, text= "Close", command=lambda: windows_events("Cerrar"))
    #Orden en matriz
    Label_Modo.grid(row=0,column=1, columnspan=2, padx=4, pady=4)
    Label_Dimension.grid(row=1,column=1, columnspan=2, padx=4, pady=4)
    Label_Infectados.grid(row=2,column=1, columnspan=2, padx=4, pady=4)
    Label_Ecuacion.grid(row=4,column=1, columnspan=2, padx=4, pady=4)
    Label_Parámetros.grid(row=5,column=1, columnspan=4, padx=4, pady=4)

    Entry_DimX.grid(row=1,column=3, columnspan=1, padx=4, pady=4)
    Entry_DimZ.grid(row=1,column=4, columnspan=1, padx=4, pady=4)
    Entry_PC.grid(row=2,column=3, columnspan=1, padx=4, pady=4)

    Combo_modo.grid(row=0,column=3, columnspan=1, padx=4, pady=4)
    Combo_ecuacion.grid(row=4,column=3, columnspan=1, padx=4, pady=4)

    Button_iniciar.grid(row=20,column=5, columnspan=1, padx=4, pady=4)
    Button_cerrar.grid(row=20,column=0, columnspan=1, padx=4, pady=4)

    #loop
    ventana.mainloop()