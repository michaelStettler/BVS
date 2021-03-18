import cv2
import numpy as np
import os


class CSVDataGen:

    def __init__(self, config, df, path, shuffle=True):
        self.start = 0
        self.batch_size = config['batch_size']
        self.df = df
        self.num_data = len(df.index)
        self.path = path
        self.shuffle = shuffle

    def reset(self):
        self.start = 0

    def generate(self):
        idx = np.arange(self.num_data)

        if shuffle:
            np.random.shuffle(idx)

        while self.start < self.num_data:
            # get batch idx
            end = min(self.start + self.batch_size, self.num_data)
            batch_idx = idx[self.start:end]

            # declare variables
            x = np.zeros((self.batch_size, 224, 224, 3))
            y = np.zeros(self.batch_size)

            for b, id in enumerate(batch_idx):
                # load img
                loc = self.df.iloc[id]
                im = cv2.imread(os.path.join(self.path, loc['subDirectory_filePath']))
                im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                # resize image
                im_rgb = cv2.resize(im_rgb, (224, 224))
                x[b, :, :, :] = im_rgb

                # fetch label
                y[b] = loc['expression']

            yield [x, y]
            self.start += self.batch_size