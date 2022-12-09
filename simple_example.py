from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import numpy as np
import os
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

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
        plt.imshow(image_batch.at(j))
    plt.show()
    plt.savefig(os.path.join("result/image/" , name))


class SimplePipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id):
        super(SimplePipeline, self).__init__(batch_size, num_threads, device_id, seed = 12)
        # self.input = ops.FileReader(file_root = image_dir, random_shuffle=True, initial_fill=16)
        # instead of path to file directory file with pairs image_name image_label_value can be provided
        # * 读取
        self.input = ops.readers.File(file_root = image_dir, file_list = txt_path)
        # * 解码
        self.decode = ops.decoders.Image(device = 'mixed', output_type = types.RGB)
        # * 随机裁剪(预处理)
        self.resizedCrop = ops.RandomResizedCrop(device="gpu", size=224, random_area=[0.08, 1.25])
    # * 设定Pipeline的执行顺序
    def define_graph(self):
        jpegs, labels = self.input()
        images = self.decode(jpegs)
        images = self.resizedCrop(images)
        return (images, labels)
# * 初始化Pipeline类
pipe = SimplePipeline(batch_size, 1, 0)
pipe.build()
# * 从pipe中读取一个batch
pipe_out = pipe.run()
print(pipe_out)
# * 可视化
images, labels = pipe_out
# 由于image已经被放到GPU上, 所以无法直接显示
show_images(batch_size, images.as_cpu(), "simple_example.png")

