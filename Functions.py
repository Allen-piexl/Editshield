from transformers import AutoTokenizer, PretrainedConfig
from torchvision import transforms
from pathlib import Path
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel
import torch
from PIL import Image, ImageOps
import os
import clip
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
torch.cuda.device_count()
cop_path = './instruct-pix2pix-main/cop_file'
model_id = "timbrooks/instruct-pix2pix"
pretrained_model_name_or_path = model_id
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",  cache_dir=cop_path,
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")

def load_data(data_dir, size=512, center_crop=False) -> torch.Tensor:
    image_transforms = transforms.Compose(
        [
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
            transforms.ToTensor(),
        ]
    )
    # images = [image_transforms(Image.open(i).convert("RGB")) for i in list(Path(data_dir).iterdir())]
    images = [image_transforms(Image.open(data_dir).convert("RGB")) ]
    images = torch.stack(images)
    return images


revision = 'fp16'
text_encoder_cls = import_model_class_from_model_name_or_path(pretrained_model_name_or_path, revision)
#%%
text_encoder = text_encoder_cls.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
text_encoder.requires_grad_(False)
unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="unet", cache_dir=cop_path, revision=revision
    )

tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="tokenizer", cache_dir=cop_path,
        revision=revision,
        use_fast=False,
    )
noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler", cache_dir=cop_path)
vae = AutoencoderKL.from_pretrained(
        pretrained_model_name_or_path, subfolder="vae", revision=revision, cache_dir=cop_path
    ).cuda()
vae.requires_grad_(False)

weight_dtype = torch.bfloat16
vae.to(device, dtype=weight_dtype)
text_encoder.to(device, dtype=weight_dtype)
unet.to(device, dtype=weight_dtype)


def get_emb(img):
    latents_1 = vae.encode(img.to(device, dtype=weight_dtype)).latent_dist.sample()
    latents = latents_1 * vae.config.scaling_factor

    # Sample noise that we'll add to the latents
    noise = torch.randn_like(latents)
    bsz = latents.shape[0]
    # Sample a random timestep for each image
    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
    timesteps = timesteps.long()

    # Add noise to the latents according to the noise magnitude at each timestep
    # (this is the forward diffusion process)
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    # Get the additional image embedding for conditioning.
    # Instead of getting a diagonal Gaussian here, we simply take the mode.
    original_image_embeds = vae.encode(img.to(device, dtype=weight_dtype)).latent_dist.sample()

    # Concatenate the `original_image_embeds` with the `noisy_latents`
    concatenated_noisy_latents = torch.cat([noisy_latents, original_image_embeds], dim=1)
    return concatenated_noisy_latents

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
