from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import numpy as np
import os
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from sklearn import preprocessing
from torch import device

txt_path = "/home/developers/liuchang/FER数据集/Oulu-CASIA/NI200cropped/10_fold_txt/0_10_fold_train.txt".format(10)
image_dir = "/home/developers/liuchang/FER数据集"
batch_size = 8


def show_images(batch_size, image_batch, name):
    columns = 4
    rows = (batch_size + 1) // (columns)
    fig = plt.figure(figsize = (32,(32 // columns) * rows))
    gs = gridspec.GridSpec(rows, columns)
    for j in range(rows*columns):
        plt.subplot(gs[j])
        plt.axis("off")
        plt.imshow(image_batch.at(j)[0,...].squeeze(), cmap="gray")
    plt.show()
    plt.savefig(os.path.join("result/image/" , name))


class ProcessPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, preprocess):
        super(ProcessPipeline, self).__init__(batch_size, num_threads, device_id, seed = 12)
        # self.input = ops.FileReader(file_root = image_dir, random_shuffle=True, initial_fill=16)
        # instead of path to file directory file with pairs image_name image_label_value can be provided
        # * 读取
        self.input = ops.readers.File(file_root = image_dir, file_list = txt_path)
        # * 解码
        self.decode = ops.decoders.Image(device = 'mixed', output_type = types.RGB)
        if preprocess == "resize":
        # * 改变尺寸
            self.preprocessor = ops.Resize(device="gpu", resize_x=60, resize_y=60, interp_type = types.INTERP_GAUSSIAN)
        if preprocess == "crop":
            self.preprocessor = ops.RandomResizedCrop(device="gpu", size = (60, 60), random_area=[0.5, 1.0])
        if preprocess == "flop":
            self.mirror = ops.random.CoinFlip(device="gpu", probability=0.5)
            mirror = fn.random.coin_flip(device="cpu", probability=0.5)
            self.preprocessor = ops.CropMirrorNormalize(scale = 0.3, device="gpu", mirror=mirror)
    # * 设定Pipeline的执行顺序
    def define_graph(self):
        jpegs, labels = self.input()
        images = self.decode(jpegs)
        images = self.preprocessor(images)
        return (images, labels)
# * 初始化Pipeline类
# preprocess = "resize"
# pipe = ProcessPipeline(batch_size, 1, 0, preprocess)
# pipe.build()
# # * 从pipe中读取一个batch
# pipe_out = pipe.run()
# print(pipe_out)
# # * 可视化
# images, labels = pipe_out
# # 由于image已经被放到GPU上, 所以无法直接显示
# show_images(batch_size, images.as_cpu(), preprocess + ".png")

preprocess = "crop"
pipe = ProcessPipeline(batch_size, 1, 0, preprocess)
pipe.build()
# * 从pipe中读取一个batch
pipe_out = pipe.run()
print(pipe_out)
# * 可视化
images, labels = pipe_out
# 由于image已经被放到GPU上, 所以无法直接显示
show_images(batch_size, images.as_cpu(), preprocess + ".png")

preprocess = "flop"
pipe = ProcessPipeline(batch_size, 1, 0, preprocess)
pipe.build()
# * 从pipe中读取一个batch
pipe_out = pipe.run()
print(pipe_out)
# * 可视化
images, labels = pipe_out
# 由于image已经被放到GPU上, 所以无法直接显示
show_images(batch_size, images.as_cpu(), preprocess + ".png")

