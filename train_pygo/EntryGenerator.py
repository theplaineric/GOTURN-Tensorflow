import os
import glob
import random
import numpy as np
from tensorflow import convert_to_tensor

class EntryGenerator(object):
    def __init__(self, images, targets, bboxgtscaled):
        self.images = images
        self.targets = targets
        self.bboxgtscaled = bboxgtscaled

    def get_next_entry(self):
        # need to convert both target and image from cv2.imread to numpy array
        index = random.randint(0, len(images))
        image = convert_to_tensor(self.images[index], dtype = tf.float32)
        target = convert_to_tensor(self.targets[index], dtype = tf.float32)
        yield ({'image': image), 'target': target, 'bbox_x1': self.bboxgtscaled[index].x1, \
            'bbox_y1': self.bboxgtscaled[index].y1, 'bbox_x2': self.bboxgtscaled[index].x2, 'bbox_y2': self.bboxgtscaled[index].y2})
