import pandas as pd
import numpy as np
import matplotlib.image as mpimg
from utils import *

import pandas as pd
import numpy as np
import matplotlib.image as mpimg
from utils import *

# get a list of random camara images. It returns the image path and the adapted steering angle pairs     
def get_data_batch(batch, data_dir = './data',  ANGLE_SHIFT = 0.3):
    camera_data =[]
    for line in batch:
        if(len(line)>0):
            center_angle = float(line[3])
            rnd = np.random.randint(0, 3)
            path = line[rnd]
            filename = data_dir + '/' + path.strip()
            if (rnd ==1):
                camera_data.append((filename, center_angle + ANGLE_SHIFT))
            elif (rnd ==2):
                camera_data.append((filename, center_angle - ANGLE_SHIFT))
            else:
                camera_data.append((filename, center_angle ))
    return camera_data

# generator for trainng gets a list (augmented) images and expected steering angles.
def generator(records, data_dir = './data', batch_size=32, augment=False):
    while True:
        num_records = len(records)
        filter_indices = np.random.randint(0, num_records, batch_size)
        batch = np.array(records)[filter_indices]
        camera_data = get_data_batch(batch, data_dir=data_dir)
        images =[]
        angles = []
        for filename, angle in camera_data:
            img = mpimg.imread(filename)
            #apply the augmentation trasforms when training
            if augment:
                img, angle = pipeline(img, float(angle))
            images.append(img)
            angles.append(float(angle))

        X = np.array(images)
        y = np.array(angles)

        # output a batch increment
        yield X, y
