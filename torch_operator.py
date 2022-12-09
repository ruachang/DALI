from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import numpy as np
import os
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import nvidia.dali.plugin.pytorch as dalitorch
import torch 
import torch.utils.dlpack as torch_dlpack
import torchvision.transforms as transforms
txt_path = "/home/developers/liuchang/FER数据集/Oulu-CASIA/NI200cropped/10_fold_txt/0_10_fold_train.txt".format(10)
image_dir = "/home/developers/liuchang/FER数据集"
batch_size = 8
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomPerspective(p=1.),
    transforms.ToTensor()
])
def perspective(t):
    return transform(t).transpose(2, 0).transpose(1, 0)

def dlpack_manipulation(dlpacks):
    tensors = [torch_dlpack.from_dlpack(dlpack) for dlpack in dlpacks]
    output = [(tensor.to(torch.float32) / 255.).sqrt() for tensor in tensors]
    output.reverse()
    return [torch_dlpack.to_dlpack(tensor) for tensor in output]

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
# * 设置基本的Pipeline
torch_function_pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=0,
                               exec_async=False, exec_pipelined=False, seed=99)

with torch_function_pipe:
    # * 使用DALI的方法进行处理
    input, _ = fn.readers.file(file_root=image_dir, file_list = txt_path, random_shuffle=True)
    im = fn.decoders.image(input, device='cpu', output_type=types.RGB)
    res = fn.resize(im, resize_x=300, resize_y=300)
    norm = fn.crop_mirror_normalize(res, std=255., mean=0.)
    # * 通过pytorch的接口导入函数
    perspective = dalitorch.fn.torch_python_function(norm, function=perspective)
    # * 通过dl_pack的接口导入函数
    sqrt_color = fn.dl_tensor_python_function(res, function=dlpack_manipulation)
    torch_function_pipe.set_outputs(perspective, sqrt_color)

torch_function_pipe.build()

x, y = torch_function_pipe.run()
show_images(batch_size,x, "pytorch_transform.png")
show_images(batch_size,y, "dl_pack_reverse.png")