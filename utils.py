import pandas as pd
import numpy as np
import cv2

def loadData(data_dir):
	columns = ['center', 'left', 'right', 'steering_angle', 'throttle', 'brake', 'speed']
	print("Dataset Columns:", columns)
	data = pd.read_csv(data_dir + '/driving_log.csv', header=None, names=columns)
	return data

def random_flip(img,steering_angle,prob=0.5):
    if (np.random.random() < prob):
        img,steering_angle= cv2.flip(img,1),-steering_angle
    return img,steering_angle

""" adjust the brightness using a random factor for all pixels"""
def random_brightness(img, median = 0.8, dev = 0.4, prob=0.5):
    if (np.random.random() < prob):
        hsv = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
        factor = median + dev * np.random.uniform(-1.0, 1.0)
        #factor = (.5+np.random.uniform())
        hsv[:,:,2] = hsv[:,:,2]*factor
        filter = hsv[:,:,2]>255
        hsv[:,:,2][filter]  = 255
        img = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)
    return img

"""sheer the  horizon by a small fraction"""
def random_shear(img, steering_angle, shear_range=200, prob =0.5):
    if np.random.random() < prob:
        h, w, ch = img.shape
        tx = np.random.randint(-shear_range, shear_range+1)
        steering_angle += tx / (h / 2) * 360 / (2 * np.pi * 25.0) / 6.0
        shear_point = [w / 2 + tx, h / 2]
        r1 = np.float32([[0, h], [w, h], [w / 2, h / 2]])
        r2 = np.float32([[0, h], [w, h], shear_point])
        transform_matrix = cv2.getAffineTransform(r1, r2)
        img = cv2.warpAffine(img, transform_matrix, (w, h), borderMode=1)
    return img, steering_angle

"""cast a random shaped shadow and overlay over the original image"""
def random_shadow(img):
    shadow = img.copy()
    h,w,ch = shadow.shape
    x1 = np.random.randint(0,int(w*0.4))
    x2 = np.random.randint(int(w*0.6),w-1)
    y1 = np.random.randint(0,int(h*0.2))
    y2 = np.random.randint(int(h*0.7),h-1)
    img = cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,0),-1)
    alpha = np.random.uniform(0.6, 0.9)
    img = cv2.addWeighted(shadow, alpha, img, 1-alpha,0,img)
    return img

"""pipe all the augmentation transforms together"""
def pipeline(img,steering_angle):
    img,steering_angle = random_flip(img,steering_angle)
    img = random_brightness(img)
    img = random_shadow(img)
    img,steering_angle = random_shear(img,steering_angle,prob=0.5)
    return img,steering_angle
