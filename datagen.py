from PIL import Image
import numpy as np
import os


class Generator:
    def __init__(self, csv, rootpath, target_len, input_size=None, batch_size=None):
        self.csv = csv
        self.rootpath = rootpath
        self.input_size = (input_size, input_size) if isinstance(input_size, int) else input_size
        self.batch_size = batch_size
        self.target_len = target_len

    def get_single_image(self, idx):
        x = np.zeros((*self.input_size, 4), dtype='float32')
        c = ['red', 'green', 'blue', 'yellow']
        for i in range(4):
            img = Image.open(os.path.join(self.rootpath, self.csv[idx][0]+f'_{c[i]}.png')).resize(self.input_size)
            x[:, :, i] = np.array(img, dtype='float32')
        x = np.rollaxis(x, 2)
        y = np.zeros(self.target_len, dtype='uint8')
        for target in self.csv[idx][1]:
            y[int(target)] = 1
        return x, y

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        if not self.batch_size:
            return self.get_single_image(idx)
