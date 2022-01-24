import numpy as np
import plotly.graph_objects as go
from matplotlib import pyplot as plt
import random
import pickle as pk

PIK = "Gauss.pk"
i = 0
Pos_Matrix = []
with open(PIK, "rb") as f:
    try:
        while True:
            #print("-------------------------------- Gauss data:",i)
            pk_file = pk.load(f)
            #print("X: "+str(pk_file))
            x = pk_file
            pk_file = pk.load(f)
            #print("Z: "+str(pk_file))
            z = pk_file
            Pos_Matrix.append([int(x), int(z)])
            #-------
            i += 1
    except EOFError:
        pass
#print(Pos_Matrix)

def gaussian_kernel (sigma):
    size = int((sigma*2)**2 + 1) #------------
    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]
    x0 = y0 = size // 2
    print("size: ",size)
    return 1/(2*np.pi*sigma**2)*np.exp(-((x-x0)**2 + (y-y0)**2)/(2*sigma**2))

def insert_basis(place, M):
    basis = gaussian_kernel(3)
    #print('BASIS: ', basis)
    w, h = basis.shape
    px, py = place
    start_x = int(px)
    start_y = int(py)
    end_x = start_x + w
    end_y = start_y + h
    #print(start_x, start_y, end_x, end_y)
    M[start_x:end_x, start_y:end_y] = basis + M[start_x:end_x, start_y:end_y]
    return M

m = 600
n = 600
M = np.zeros((m, n))
x = np.arange(0, m, step = 1)
y = np.arange(0, n, step = 1)
#---
contador_segundos = 0
contador_segundos2 = 0
contador_esferas = 0
x_test = 0
y_test = 0
plt.ion()
while contador_esferas<len(Pos_Matrix):
    try:
        x_test = int(Pos_Matrix[contador_esferas][0])
        y_test = int(Pos_Matrix[contador_esferas][1])
        print('i', contador_esferas, 'x: ', x_test, 'y: ', y_test)
        M = insert_basis((x_test, y_test), M)
        plt.pause(1)
        plt.imshow(M)
        contador_esferas += 1
    except EOFError:
        pass

fig = go.Figure(go.Surface(
    contours = {
        "x": {"show": True, "start": 1.5, "end": 2, "size": 0.04, "color":"white"},
        "z": {"show": True, "start": 0.5, "end": 0.8, "size": 0.05}
    },
    x = x,
    y = y,
    z = M))
fig.update_layout(
        scene = {
            "xaxis": {"nticks": 20},
            "zaxis": {"nticks": 5},
            'camera_eye': {"x": 0, "y": -1, "z": 0.5},
            "aspectratio": {"x": 1, "y": 1, "z": 0.2}
        })

fig.show()