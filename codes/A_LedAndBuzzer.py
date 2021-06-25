import Jetson.GPIO as GPIO
import time

toDisplay="4" 
delay = 0.005 # delay between digits refresh

GPIO.setmode(GPIO.BOARD)

display_leds = [12,11,13,15,16,7,19,21,23,22,24,26]
'''
[12,11,13,15,16, 7,19,21,23,22,24,26] jetson
[1 ,2 , 3, 4, 5, 6, 7, 8, 9,10,11,12] display
'''
a=display_leds[10]
b=display_leds[6]
c=display_leds[3]
d=display_leds[1]
e=display_leds[0]
f=display_leds[9]
g=display_leds[4]

dig_point=display_leds[2]
dig1=display_leds[11]
dig2=display_leds[8]
dig3=display_leds[7]
dig4=display_leds[5]
dig2=display_leds[8]

selDigit = [dig4]
display_list = [a,b,c,d,e,f,g] 
digitDP = dig_point
#---------------------------------------------------Configuracion Pins
GPIO.setwarnings(False)
for pin in display_list:
  GPIO.setup(pin,GPIO.OUT) 
for pin in selDigit:
  GPIO.setup(pin,GPIO.OUT)
GPIO.setup(digitDP,GPIO.OUT)
GPIO.setwarnings(True)

arrSeg = [[1,1,1,1,1,0,0],\
          [0,1,1,0,0,0,0],\
          [1,1,0,1,1,0,1],\
          [1,1,1,1,0,0,1],\
          [0,1,1,0,0,1,1],\
          [1,0,1,1,0,1,1],\
          [1,0,1,1,1,1,1],\
          [1,1,1,0,0,0,0],\
          [1,1,1,1,1,1,1],\
          [1,1,1,1,0,1,1]]



GPIO.output(digitDP,0) # DOT pin

def showDisplay(digit):
 for i in range(0, 1): #loop on 4 digits selectors (from 0 to 3 included)
  sel = [0]
  sel[i] = 1
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

try:
 while True:
  showDisplay(splitToDisplay(toDisplay))
except KeyboardInterrupt:
 print('interrupted!')
 GPIO.cleanup()
