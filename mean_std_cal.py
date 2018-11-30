import numpy as np
from PIL import Image
import os


# [418428 401479 621813 639736] [1082904  681422 1014570  992136] / 31072
def get_mean_std():
    path = '/media/palm/data/Human Protein Atlas/train'
    files = sorted(os.listdir(path))
    mean = np.zeros(4, dtype='uint64')
    std = np.zeros(4, dtype='uint64')
    i = 0
    for image in files:
        x = Image.open(os.path.join(path, image))
        x = np.array(x)
        mean[i % 4] += np.mean(x)
        std[i % 4] += np.std(x)
        i += 1
    return mean, std


def normalize(x):
    x = x.astype('float32')
    x -= np.array([13.46640062, 12.92092559, 20.01200438, 20.58882595], dtype='float32')
    x /= np.array([34.85144181, 21.93041967, 32.65222709, 31.93022657], dtype='float32')
    return x
