import numpy as np
import cv2

i = 1
a = cv2.imread('TRAIN'+str(i)+'.png')*255
b = cv2.imread('TEST'+ str(i-1)+'.png')*255

c = a - b
print(np.max(np.abs(c)))