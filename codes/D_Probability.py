import math
import numpy as np

def Well_Riley(i, t, M):
    q = 12
    p = 0.52
    Q = 10
    Exponente = -(i*q*p*t)/(360*Q*(1+M))
    Probabilidad = (1-(2.71828 ** Exponente))*100
    return Probabilidad

def Well_Riley_Mod(i, M): #This modification assumes that data will be taken every minute
    q = 12
    p = 0.52
    Q = 10
    Exponente = -(i*q*p)/(360*Q*(1+M))
    return Exponente

N=1 #numero de personas
M=1 #uso de mascarilla
E=0 #inicializacion del contador del exponente
t=0 #inicializacion del tiempo

while t <= 120:
    t+=1
    E += Well_Riley_Mod(N, M)
    P = (1-(2.71828 ** E))*100    
    P2=Well_Riley(N, t, M)
    print ("minutos:",t," Wells Riley Mod:",round(P,2)," Wells Riley:", round(P2,2))
    