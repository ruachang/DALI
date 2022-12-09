from nvidia.dali import pipeline_def, Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types

txt_path = "/home/developers/liuchang/FER数据集/Oulu-CASIA/NI200cropped/10_fold_txt/0_10_fold_train.txt".format(10)
image_dir = "/home/developers/liuchang/FER数据集"
@pipeline_def
def caffe_pipeline(num_gpus):
    device_id = Pipeline.current().device_id
    jpegs, labels = fn.readers.file(
        name='Reader', file_root = image_dir, file_list = txt_path, random_shuffle=True, shard_id=device_id, num_shards=num_gpus)
    images = fn.decoders.image(jpegs, device='mixed')
    images = fn.resize(
        images,
        resize_shorter=fn.random.uniform(range=(256, 480)),
        interp_type=types.INTERP_LINEAR)
    images = fn.crop_mirror_normalize(
            images,
            crop_pos_x=fn.random.uniform(range=(0.0, 1.0)),
            crop_pos_y=fn.random.uniform(range=(0.0, 1.0)),
            dtype=types.FLOAT,
            crop=(227, 227),
            mean=[128., 128., 128.],
            std=[1., 1., 1.])

    return images, labels

import numpy as np
from nvidia.dali.plugin.pytorch import DALIGenericIterator


label_range = (0, 999)
pipes = [caffe_pipeline(
    batch_size=8, num_threads=2, device_id=device_id, num_gpus=2) for device_id in range(2)]
# * 多个管道
for pipe in pipes:
    pipe.build()
# * 作为PyTorch的迭代器读入
dali_iter = DALIGenericIterator(pipes, ['data', 'label'], reader_name='Reader')

for i, data in enumerate(dali_iter):
    # Testing correctness of labels
    # * 这里的维度和pipe的数量有关系
    for d in data:
        label = d["label"]
        image = d["data"]
        ## labels need to be integers
        assert(np.equal(np.mod(label, 1), 0).all())
        ## labels need to be in range pipe_name[2]
        assert((label >= label_range[0]).all())
        assert((label <= label_range[1]).all())

print("OK")