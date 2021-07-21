import math
import numpy as np

P_dist=0
x=0
while x <= 11:
    if x > 1 and x <10:
        P_dist = -0.2889*x + 2.8889
    elif  x >= 0 and x <= 1:
        P_dist = -10.2*x + 12.8
    else:
        P_dist = 0
    print ("Distance:",x," Probability:",round(P_dist,2))
    x+=1    