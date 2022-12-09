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


class SharedPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, shard_id, shard_nuums):
        super(SharedPipeline, self).__init__(batch_size, num_threads, device_id, seed = 12)
        # self.input = ops.FileReader(file_root = image_dir, random_shuffle=True, initial_fill=16)
        # instead of path to file directory file with pairs image_name image_label_value can be provided
        # * 读取
        self.input = ops.readers.File(file_root = image_dir, file_list = txt_path, shard_id = shard_id, num_shards = shard_nuums)
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
pipe1 = SharedPipeline(batch_size, num_threads=2, device_id=0, shard_id=0, shard_nuums=2)
pipe1.build()
# * 从pipe中读取一个batch
pipe_out1 = pipe1.run()
# * 可视化
images, labels = pipe_out1
# # 由于image已经被放到GPU上, 所以无法直接显示
# show_images(batch_size, images.as_cpu(), "simple_example.png")

# * 初始化Pipeline类
pipe2 = SharedPipeline(batch_size, num_threads=2, device_id=1, shard_id=1, shard_nuums=2)
pipe2.build()
# * 从pipe中读取一个batch
pipe_out2 = pipe2.run()

images, labels = pipe_out2