import numpy as np

class FeaturePoint:
    def __init__(self, x, y, orientation, level, descriptor):
        self.x = x # with respect to first image level
        self.y = y
        self.orientation = orientation #[cos, sin]
        self.level = level #detected at which level
        self.descriptor = descriptor 
