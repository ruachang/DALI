import numpy as np
from random import shuffle
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import matplotlib.pyplot as plt
from PIL import Image

# * 构建自建类用来读取