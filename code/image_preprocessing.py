import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob

from skimage.color import rgb2lab
from skimage.transform import resize
from collections import namedtuple

import warnings
warnings.filterwarnings(action = 'ignore', category = FutureWarning)


# 전처리
dataset = namedtuple('Dataset', ['x', 'y'])
RESIZED_IMAGE = (512, 512)
folder = ['Heart', 'Oblong', 'Oval', 'Round', 'Square']

# 1. 이미지 크기 조절 512*512
# 2. 이미지 색상 회색조로 변경
# 3. label one-hot encoding
def to_tf_format(imgs):
    return np.stack([img[:,:, np.newaxis] for img in imgs],
                    axis=0).astype(np.float32)

def read_img(path, img_labels, resize_to):
    images = []
    labels = []
    for i in range(len(img_labels)):
        full_path = path + '/' + img_labels[i] + '/'
        
        for img_name in glob.glob(full_path + '*.jpg'):
            img = plt.imread(img_name).astype(np.float32)
            img = rgb2lab(img/255.0)[:, :, 0]
            img = resize(img, resize_to, mode='reflect')
            
            label = np.zeros((len(labels), ), dtype=np.float32)
            label[i] = 1.0
            
            images.append(img.astype(np.float32))
            labels.append(label)
    
    return dataset(x=to_tf_format(images), y=np.array(labels))


ds = read_img('../data/images', folder, RESIZED_IMAGE)

df = pd.DataFrame(ds)
df.to_csv('../data', header=False, index=False)