import torch
import sys

class Labels:
    # label for real (non-watermarked)
    REAL = 'original'
    # label for fake (watermarked)
    FAKE = 'watermarked'
    
class ScoreColumns:
    # name for score csv file

    # column for watermark detection score
    SCORE = 'score'
    # column for label
    LABEL = 'label'
    # column for filename
    FILENAME = 'name'
    # column for augmentation
    AUGMENTATION = 'augmentation'

    
class add_path():
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        sys.path.insert(0, self.path)

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            sys.path.remove(self.path)
        except ValueError:
            pass


def return_true():

    return True
