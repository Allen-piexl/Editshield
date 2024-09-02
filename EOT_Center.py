#%%
import sys
import numpy as np
from omegaconf import OmegaConf
import PIL
from PIL import Image
from einops import rearrange
import ssl
from tqdm import tqdm
import time
import torch
import os
from PIL import Image, ImageOps
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from diffusers import DiffusionPipeline
import copy
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from torch import optim
import json
import random
random.seed(333)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
model_id = "timbrooks/instruct-pix2pix"
pretrained_model_name_or_path = model_id
torch.cuda.device_count()
from torchvision import transforms
from pathlib import Path
from Functions import *

import pandas as pd
#%%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_path = './instruct-pix2pix-main/diffuser_cache'
cop_path = './instruct-pix2pix-main/cop_file'
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained('timbrooks/instruct-pix2pix', torch_dtype=torch.float16,
                                                 safety_checker=None, cache_dir=model_path, local_files_only=True)
pipe.to("cuda")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)



#%%
pgd_alpha = 1 / 255
pgd_eps = 0.05
max_steps = 30
center_crop = False

#%%
# i = 0
def center_crop(images, new_height, new_width):
    # Calculate cropping start and end points
    _, _, height, width = images.shape
    startx = width // 2 - (new_width // 2)
    starty = height // 2 - (new_height // 2)
    endx = startx + new_width
    endy = starty + new_height

    # Crop and return
    return images[:, :, starty:endy, startx:endx]


if not os.path.exists('./instruct-pix2pix-main/002_Data/EOT-C/Gen'):
    os.mkdir('./instruct-pix2pix-main/002_Data/EOT-C/Gen')
if not os.path.exists('./instruct-pix2pix-main/002_Data/EOT-C/Adv'):
    os.mkdir('./instruct-pix2pix-main/002_Data/EOT-C/Adv')
if not os.path.exists('./instruct-pix2pix-main/002_Data/EOT-C/AdvGen'):
    os.mkdir('./instruct-pix2pix-main/002_Data/EOT-C/AdvGen')
if not os.path.exists('./instruct-pix2pix-main/002_Data/EOT-C/Ori'):
    os.mkdir('./instruct-pix2pix-main/002_Data/EOT-C/Ori')
if not os.path.exists('./instruct-pix2pix-main/002_Data/EOT-C/EXCEL'):
    os.mkdir('./instruct-pix2pix-main/002_Data/EOT-C/EXCEL')

root_path = './instruct-pix2pix-main/train_data'
train_data_path = os.listdir('./instruct-pix2pix-main/train_data')

for i in range(len(train_data_path)):
    print(train_data_path[i])
    save_ori = os.path.join('./instruct-pix2pix-main/002_Data/EOT-C/Ori', train_data_path[i])
    if not os.path.exists(save_ori):
        os.mkdir(save_ori)
    save_gen = os.path.join('./instruct-pix2pix-main/002_Data/EOT-C/Gen', train_data_path[i])
    if not os.path.exists(save_gen):
        os.mkdir(save_gen)
    save_adv = os.path.join('./instruct-pix2pix-main/002_Data/EOT-C/Adv', train_data_path[i])
    if not os.path.exists(save_adv):
        os.mkdir(save_adv)
    save_advgen = os.path.join('./instruct-pix2pix-main/002_Data/EOT-C/AdvGen', train_data_path[i])
    if not os.path.exists(save_advgen):
        os.mkdir(save_advgen)
    save_result = os.path.join('./instruct-pix2pix-main/002_Data/EOT-C/EXCEL', train_data_path[i])
    if not os.path.exists(save_result):
        os.mkdir(save_result)
    print('Make Dir Done!')


    image_path = os.path.join(root_path, train_data_path[i])
    image_list = os.listdir(image_path)
    resolution = 512
    with open(os.path.join(image_path, 'prompt.json'), 'r', encoding='utf-8') as f:
        load_json = json.load(f)
    prompt = load_json['edit']
    name_list = []
    sim_image_bef_list = []
    sim_image_aft_list = []
    sim_image_adv_list = []
    angle = 5
    for j in range(len(image_list)):
        if image_list[j].endswith('_0.jpg'):
            # print('hh')
            input_path = os.path.join(image_path, image_list[j])
            name_list.append(image_list[j])
            perturbed_data = load_data(input_path, resolution, center_crop=False)  # 初始img  可更新
            resolution = min(perturbed_data.size(2), perturbed_data.size(3))  # Assuming square cropping
            perturbed_data = center_crop(perturbed_data, resolution, resolution)

            tgt_data = load_data(input_path, resolution, center_crop=False)

            original_data = perturbed_data.clone()  # 初始img
            aaa = original_data.detach().cpu().numpy()[0]
            plt.imsave(os.path.join(save_ori, image_list[j]), aaa.transpose(1, 2, 0))
            generator = torch.Generator("cuda").manual_seed(33)
            images = \
            pipe(prompt, image=Image.open(input_path).resize((512, 512)), num_inference_steps=100, image_guidance_scale=1.2,
                 generator=generator).images[0]
            images.save(os.path.join(save_gen, image_list[j]))
            original_images = original_data
            perturbed_images = perturbed_data.detach().clone()
            tgt_images = tgt_data.detach().clone()
            tgt_emb = get_emb(tgt_images).detach().clone()

            optimizer = optim.Adam([perturbed_images])
            for step in range(max_steps):
                perturbed_images.requires_grad = True
                img_emb = get_emb(perturbed_images)
                optimizer.zero_grad()

                vae.zero_grad()
                loss = -F.mse_loss(img_emb.float(), tgt_emb.float())

                loss.backward()
                optimizer.step()

                if step % 10 == 0:
                    print(f"PGD loss - step {step}, loss: {loss.detach().item()}")
            noised_imgs = perturbed_images.detach().cpu().numpy()[0]
            plt.imsave(os.path.join(save_adv, image_list[j]),
                       np.clip(noised_imgs.transpose(1, 2, 0), 0, 1))

            generator = torch.Generator("cuda").manual_seed(33)
            images = \
            pipe(prompt, image=Image.fromarray(np.uint8(noised_imgs.transpose(1, 2, 0) * 255)), num_inference_steps=100,
                 image_guidance_scale=1.2, generator=generator).images[0]

            images.save(os.path.join(save_advgen, image_list[j]))

            x = np.array(Image.open(os.path.join(save_ori, image_list[j])).resize(
                (512, 512))) / 255  # benign
            x_adv = np.array(
                Image.open(os.path.join(save_adv, image_list[j])).resize(
                    (512, 512))) / 255  # benign
            x_gen = np.array(Image.open(os.path.join(save_gen, image_list[j])).resize(
                (512, 512))) / 255  # ori_xg
            x_gen_attack = np.array(
                Image.open(os.path.join(save_advgen, image_list[j])).resize(
                    (512, 512))) / 255

            clip_similarity = ClipSimilarity().cuda()
            image_features_benign = clip_similarity.encode_image(
                image=torch.tensor(x.transpose(2, 0, 1)).unsqueeze(0).to(device))
            image_features_gen = clip_similarity.encode_image(
                image=torch.tensor(x_gen.transpose(2, 0, 1)).unsqueeze(0).to(device))
            image_feature_adv = clip_similarity.encode_image(
                image=torch.tensor(x_adv.transpose(2, 0, 1)).unsqueeze(0).to(device))
            image_features_attack = clip_similarity.encode_image(
                image=torch.tensor(x_gen_attack.transpose(2, 0, 1)).unsqueeze(0).to(device))
            sim_image_bef = F.cosine_similarity(image_features_benign, image_features_gen)[0]
            sim_image_aft = F.cosine_similarity(image_features_benign, image_features_attack)[0]
            sim_image_adv = F.cosine_similarity(image_features_benign, image_feature_adv)[0]
            sim_image_bef_list.append(sim_image_bef.detach().cpu().numpy())
            sim_image_aft_list.append(sim_image_aft.detach().cpu().numpy())
            sim_image_adv_list.append(sim_image_adv.detach().cpu().numpy())


        else:
            continue
    data = {'file_name': name_list, 'sim_image_bef':sim_image_bef_list, 'sim_image_aft':sim_image_aft_list, 'sim_image_adv':sim_image_adv_list}
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(save_result, 'result.csv'), index=False)