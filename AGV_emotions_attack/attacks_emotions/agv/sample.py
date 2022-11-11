import matplotlib.pyplot as plt
import numpy as np

a = plt.imread('TRAIN.png')*255
b = plt.imread('TEST.png')*255
c = a - b
print(np.max(np.abs(c)))