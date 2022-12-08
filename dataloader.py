from typing import Iterator
import numpy as np
from random import shuffle
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import matplotlib.pyplot as plt
from PIL import Image
import os

import nvidia.dali.types as types
txt_path = "/home/developers/liuchang/FER数据集/Oulu-CASIA/NI200cropped/10_fold_txt/0_10_fold_train.txt".format(10)
image_dir = "/home/developers/liuchang/FER数据集"
batch_size = 8

# * 构建自建类用来读取
class ExternalInputIterator(object):
    def __init__(self, batch_size) -> None:
        self.txt_path = txt_path
        self.image_dir =image_dir
        self.batch_size = batch_size
        with open(self.txt_path, 'r') as f:
            self.files = [line.rstrip() for line in f if line != '']
        shuffle(self.files)
    def __iter__(self):
        self.i = 0
        self.n = len(self.files)
        return self
    def __next__(self):
        batch = []
        labels = []
        for _ in range(self.batch_size):
            # 取决于txt文件的样子
            file_name , label = self.files[self.i].split(" ")
            file_path = os.path.join(self.image_dir , file_name)
            f = open(file_path, 'rb')
            batch.append(np.frombuffer(f.read(), dtype=np.uint8))
            labels.append(np.array([label], dtype=np.float32))
            self.i = (self.i + 1) % self.n 
        return (batch, labels)

# * 初始化类
iterator = ExternalInputIterator(batch_size)
pipe = Pipeline(batch_size = batch_size, num_threads=2, device_id=0)
with pipe:
    #  * 定义导入
    jpegs, labels = fn.external_source(source=iterator, num_outputs=2)
    decode = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)
    enhance = fn.brightness_contrast(decode, contrast=2)
    # * 作为输出
    pipe.set_outputs(enhance, labels)

pipe.build()
pipe_out = pipe.run()
batch_cpu = pipe_out[0].as_cpu()
labels_cpu = pipe_out[1]

from simple_example import show_images
show_images(batch_size,  batch_cpu, "load_cpu.png")

# import numpy as np
# from random import shuffle
# from nvidia.dali.pipeline import Pipeline
# import nvidia.dali.fn as fn
# import matplotlib.pyplot as plt
import cupy as cp
import imageio

root_path = '/home/developers/liuchang/FER数据集'
# batch_size = 4

class CustomizeInputGpuIterator(object):
    def __init__(self, images_dir, batch_size):
        self.images_dir = images_dir
        self.batch_size = batch_size
        with open("/home/developers/liuchang/FER数据集/Oulu-CASIA/NI200cropped/10_fold_txt/0_10_fold_train.txt".format(10), 'r') as f:
            self.files = [line.rstrip() for line in f if line != '']
        shuffle(self.files)

    def __iter__(self):
        self.idx = 0
        self.length = len(self.files)
        return self

    def __next__(self):
        batch = []
        labels = []
        for _ in range(self.batch_size):
            jpeg_filename, label = self.files[self.idx].split(' ')
            # * 必须直接CPU解码
            im = imageio.imread(os.path.join(self.images_dir , jpeg_filename))
            im = cp.asarray(im)
            im = im * 0.6
            batch.append(im.astype(cp.uint8))
            labels.append(cp.array([label], dtype = np.uint8))
            self.idx = (self.idx + 1) % self.length
        return (batch, labels)

eii_gpu = CustomizeInputGpuIterator(root_path, batch_size)
print(type(next(iter(eii_gpu))[0][0]))

pipe_gpu = Pipeline(batch_size=batch_size, num_threads=2, device_id=0)
with pipe_gpu:
    images, labels = fn.external_source(source=eii_gpu, num_outputs=2, device="gpu")
    enhance = fn.brightness_contrast(images, contrast=2)
    pipe_gpu.set_outputs(enhance, labels)

pipe_gpu.build()

pipe_out_gpu = pipe_gpu.run()
batch_gpu = pipe_out_gpu[0].as_cpu()
labels_gpu = pipe_out_gpu[1].as_cpu()

show_images(batch_size, batch_gpu, "load_gpu.png")
# img = batch_gpu.at(2)
# print(img.shape)
# plt.axis('off')
# plt.imshow(img)
# plt.show()
