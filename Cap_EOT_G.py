#%%
import sys
import numpy as np
import PIL
from PIL import Image
from einops import rearrange
import ssl
from tqdm import tqdm
import time
import torch
import os

import copy
import torch.nn.functional as F
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from torch import optim
import json
import pandas as pd
import torch.nn as nn
from pathlib import Path
import random
random.seed(333)
#%%
python_root = "python -m ./LLaVA/llava/serve.cli --model-path ./LLaVA-Lightning-MPT-7B-preview  --image-file "
root_path = './instruct-pix2pix-main/train_data'
train_data_path = os.listdir('./instruct-pix2pix-main/train_data')

for i in range(len(train_data_path)):
    save_ori = os.path.join('./instruct-pix2pix-main/002_Data/EOT-G/Ori', train_data_path[i])
    save_gen = os.path.join('./instruct-pix2pix-main/002_Data/EOT-G/Gen', train_data_path[i])
    save_adv = os.path.join('./instruct-pix2pix-main/002_Data/EOT-G/Adv', train_data_path[i])
    save_advgen = os.path.join('./instruct-pix2pix-main/002_Data/EOT-G/AdvGen', train_data_path[i])
    save_result = os.path.join('./instruct-pix2pix-main/002_Data/EOT-G/EXCEL', train_data_path[i])


    image_path = os.path.join(root_path, train_data_path[i])
    image_list = os.listdir(image_path)
    name_list = []
    sim_direction_bef_list = []
    sim_direction_aft_list = []
    for j in range(len(image_list)):
        if image_list[j].endswith('_0.jpg'):
            name_list.append(image_list[j])

            x_path = os.path.join(save_ori, image_list[j])
            x_adv_path = os.path.join(save_adv, image_list[j])
            x_gen_path = os.path.join(save_gen, image_list[j])
            x_gen_attack_path = os.path.join(save_advgen, image_list[j])

            os.system( "python ./LLaVA/llava/serve/cli.py --model-path "
                       "./LLaVA-Lightning-MPT-7B-preview  --image-file {} --save_path {}".format(x_path, x_path))

            os.system( "python ./LLaVA/llava/serve/cli.py --model-path "
                       "./LLaVA-Lightning-MPT-7B-preview  --image-file {} --save_path {}".format(x_adv_path, x_adv_path))

            os.system( "python ./LLaVA/llava/serve/cli.py --model-path "
                       "./LLaVA-Lightning-MPT-7B-preview  --image-file {} --save_path {}".format(x_gen_path, x_gen_path))

            os.system( "python ./LLaVA/llava/serve/cli.py --model-path "
                       "./LLaVA-Lightning-MPT-7B-preview  --image-file {} --save_path {}".format(x_gen_attack_path, x_gen_attack_path))
