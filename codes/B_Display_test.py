import sys
import Jetson.GPIO as GPIO
import time

GPIO.setmode(GPIO.BOARD)

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

try:
  pi=0
  while True:
    toDisplay=str(pi)
    showDisplay(splitToDisplay(toDisplay))
    print(pi)
    pi+=1        
    time.sleep(2)

except KeyboardInterrupt:
 print('interrupted!')
 GPIO.cleanup()
sys.exit()