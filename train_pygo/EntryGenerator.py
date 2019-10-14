import os
import glob
import random

class EntryGenerator(object):
    def __init__(self, images, targets, bboxgtscaled):
        self.images = images
        self.targets = targets
        self.bboxgtscaled = bboxgtscaled

    def get_next_entry(self):
        index = random.randint(0, len(images))
        yield ({'image': self.images[index], 'target': self.targets[index], 'bbox': self.bboxgtscaled[index]})
