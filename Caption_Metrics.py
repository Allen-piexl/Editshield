
#%%
import sys
sys.path.append("./instruct-pix2pix-main")
sys.path.append("./LLaVA/llava")
sys.path.append("./instruct-pix2pix-main/001_Code")
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
import clip
import random
random.seed(333)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import argparse
parser = argparse.ArgumentParser(
    description='Dataset size')
parser.add_argument('--number', type=float)
parser.add_argument('--method', type=str)
args = parser.parse_args()
class ClipSimilarity(nn.Module):
    def __init__(self, name: str = "ViT-L/14"):
        super().__init__()
        assert name in ("RN50", "RN101", "RN50x4", "RN50x16", "RN50x64", "ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px")  # fmt: skip
        self.size = {"RN50x4": 288, "RN50x16": 384, "RN50x64": 448, "ViT-L/14@336px": 336}.get(name, 224)

        self.model, _ = clip.load("./instruct-pix2pix-main/clip-vit-large-patch14/ViT-L-14.pt", device="cpu", download_root="./")
        self.model.eval().requires_grad_(False)

        self.register_buffer("mean", torch.tensor((0.48145466, 0.4578275, 0.40821073)))
        self.register_buffer("std", torch.tensor((0.26862954, 0.26130258, 0.27577711)))

    def encode_text(self, text: list):
        text = clip.tokenize(text, truncate=True).to(next(self.parameters()).device)
        text_features = self.model.encode_text(text)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        return text_features

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:  # Input images in range [0, 1].
        image = F.interpolate(image.float(), size=self.size, mode="bicubic", align_corners=False)
        image = image - rearrange(self.mean, "c -> 1 c 1 1")
        image = image / rearrange(self.std, "c -> 1 c 1 1")
        image_features = self.model.encode_image(image)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        return image_features

    def forward(
        self, image_0: torch.Tensor, image_1: torch.Tensor, text_0: list, text_1: list):
        image_features_0 = self.encode_image(image_0)
        image_features_1 = self.encode_image(image_1)
        text_features_0 = self.encode_text(text_0)
        text_features_1 = self.encode_text(text_1)
        sim_0 = F.cosine_similarity(image_features_0, text_features_0)
        sim_1 = F.cosine_similarity(image_features_1, text_features_1)
        sim_direction = F.cosine_similarity(image_features_1 - image_features_0, text_features_1 - text_features_0)
        sim_image = F.cosine_similarity(image_features_0, image_features_1)
        return sim_0, sim_1, sim_direction, sim_image

#%%
python_root = "python -m ./LLaVA/llava/serve.cli --model-path ./LLaVA-Lightning-MPT-7B-preview  --image-file "
root_path = './instruct-pix2pix-main/train_data'
train_data_path = os.listdir('./instruct-pix2pix-main/train_data')

sum_count = args.number

image_path = './instruct-pix2pix-main/002_Data/{}/{}/Ori'.format(parser.method,sum_count)
image_list = os.listdir(image_path)
sim_direction_bef_list = []
sim_direction_aft_list = []
name_list = []
save_result = './instruct-pix2pix-main/002_Data/{}/{}/EXCEL'.format(parser.method,sum_count)
cc = 0
#%%
for j in tqdm(range(len(image_list))):
    if not image_list[j].endswith('.csv'):
        name_list.append(image_list[j])

        x_path = os.path.join('./instruct-pix2pix-main/002_Data/{}/{}/Ori'.format(parser.method, sum_count), image_list[j])
        x_adv_path = os.path.join('./instruct-pix2pix-main/002_Data/{}/{}/Adv'.format(parser.method,sum_count), image_list[j])
        x_gen_path = os.path.join('./instruct-pix2pix-main/002_Data/{}/{}/Gen'.format(parser.method,sum_count), image_list[j])
        x_gen_attack_path = os.path.join('./instruct-pix2pix-main/002_Data/{}/{}/AdvGen'.format(parser.method,sum_count), image_list[j])
        x = np.array(Image.open(x_path).resize(
            (512, 512))) / 255  # benign
        x_adv = np.array(
            Image.open(x_adv_path).resize(
                (512, 512))) / 255  # benign
        x_gen = np.array(Image.open(x_gen_path).resize(
            (512, 512))) / 255  # ori_xg
        x_gen_attack = np.array(
            Image.open(x_gen_attack_path).resize(
                (512, 512))) / 255

        benign_caption_ = pd.read_csv(x_path +'.csv')['0']
        benign_caption = benign_caption_[0]
        adv_caption_ = pd.read_csv(x_adv_path + '.csv')['0']
        adv_caption = adv_caption_[0]

        benign_out_caption_ = pd.read_csv(x_gen_path + '.csv')['0']
        benign_out_caption = benign_out_caption_[0]
        adv_out_caption1_ = pd.read_csv(x_gen_attack_path+ '.csv')['0']
        adv_out_caption1 = adv_out_caption1_[0]


        clip_similarity = ClipSimilarity().cuda()
        image_features_benign = clip_similarity.encode_image(
            image=torch.tensor(x.transpose(2, 0, 1)).unsqueeze(0).to(device))
        image_features_gen = clip_similarity.encode_image(
            image=torch.tensor(x_gen.transpose(2, 0, 1)).unsqueeze(0).to(device))
        image_feature_adv = clip_similarity.encode_image(
            image=torch.tensor(x_adv.transpose(2, 0, 1)).unsqueeze(0).to(device))
        image_features_attack = clip_similarity.encode_image(
            image=torch.tensor(x_gen_attack.transpose(2, 0, 1)).unsqueeze(0).to(device))
        text_f_benign = clip_similarity.encode_text([benign_caption])
        text_f_benign_out = clip_similarity.encode_text([benign_out_caption])
        text_f_adv = clip_similarity.encode_text([adv_caption])
        text_f_adv_out = clip_similarity.encode_text([adv_out_caption1])

        sim_0 = F.cosine_similarity(image_features_benign, text_f_benign)
        sim_1 = F.cosine_similarity(image_features_gen, text_f_benign_out)
        sim_direction_bef = F.cosine_similarity(image_features_gen - image_features_benign,
                                                text_f_benign_out - text_f_benign)

        sim_direction_aft = F.cosine_similarity(image_features_attack - image_features_benign,
                                                text_f_adv_out - text_f_benign)
        sim_direction_bef_list.append(sim_direction_bef.detach().cpu().numpy()[0])
        sim_direction_aft_list.append(sim_direction_aft.detach().cpu().numpy()[0])
        cc +=1
        # if cc >150:
        #     break
data = {'file_name': name_list, 'sim_direction_bef':sim_direction_bef_list, 'sim_direction_aft':sim_direction_aft_list}
df = pd.DataFrame(data)
df.to_csv(os.path.join(save_result, 'result_llama.csv'), index=False)
