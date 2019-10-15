import os
import glob
import random
import numpy as np

class EntryGenerator(object):
    def __init__(self, images, targets, bboxgtscaled):
        self.images = images
        self.targets = targets
        self.bboxgtscaled = bboxgtscaled

    def get_next_entry(self):
        # need to convert both target and image from cv2.imread to numpy array
        index = random.randint(0, len(images))
        yield ({'image': np.asarray(self.images[index]), 'target': np.asarray(self.targets[index]), 'bbox_x1': self.bboxgtscaled[index].x1, \
            'bbox_y1': self.bboxgtscaled[index].y1, 'bbox_x2': self.bboxgtscaled[index].x2, 'bbox_y2': self.bboxgtscaled[index].y2})
