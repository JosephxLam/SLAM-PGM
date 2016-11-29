import GeneratePoints
import math
import numpy as np

class Robot():
    def __init__(self, xMax, yMax):
        self.xMax = xMax
        self.yMax = yMax
        self.Position = GeneratePoints(1,xMax,yMax)
        self.direction = 2 * math.pi * np.random.rand()
        