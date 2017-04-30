import numpy as np

class FeaturePoint:
    def __init__(self, x, y, orientation, level, descriptor):
        self.x = x
        self.y = y
        self.orientation = orientation #[cos, sin]
        self.level = level #detected at which level
        self.descriptor = descriptor 
