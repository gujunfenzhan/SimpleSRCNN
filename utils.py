from pathlib import Path
import random

import torch
from PIL.Image import Resampling
from PIL import Image
from torch import Tensor
from torchvision import transforms


class ImageUtils:
    """
    不调整图像宽高，调整图像清晰度
    """

    @staticmethod
    def scale_definition(image, scale):
        original_width, original_height = image.size
        scaled_width = int(original_width * scale)
        scaled_height = int(original_height * scale)
        lr_image = image.resize((scaled_width, scaled_height), Resampling.BICUBIC)
        ld_image = lr_image.resize((original_width, original_height), Resampling.BICUBIC)
        return ld_image

    '''
    图像裁剪成一个个的图像块
    输入:[C, W, H]
    '''

    @staticmethod
    def crop_image_to_patches(image: Tensor, *, patch_size=64, patch_stride=64) -> list[Tensor]:
        patches = []
        channel, width, height = image.shape
        for i in range(0, width - patch_size + 1, patch_stride):
            for j in range(0, height - patch_size + 1, patch_stride):
                patches.append(image[:, i:i + patch_size, j:j + patch_size])
        return patches

    @staticmethod
    def images_to_patches_by_root_path(path: Path, minimum_scale, maximum_scale, patch_size, patch_stride):
        transform_to_tensor = transforms.ToTensor()
        image_paths = path.glob("*.png")
        hd_patches = []
        ld_patches = []
        for image_path in image_paths:
            hd_image = Image.open(image_path).convert("YCbCr")
            # 图像降清晰处理
            scale = random.random() * (maximum_scale - minimum_scale) + minimum_scale
            ld_image = ImageUtils.scale_definition(hd_image, scale)

            # 转为tensor并分离Y层
            hd_image = transform_to_tensor(hd_image)[0].unsqueeze(0)
            ld_image = transform_to_tensor(ld_image)[0].unsqueeze(0)
            # 图像裁剪成块
            hd_patches.extend(ImageUtils.crop_image_to_patches(hd_image, patch_size=patch_size,
                                                               patch_stride=patch_stride))
            ld_patches.extend(ImageUtils.crop_image_to_patches(ld_image, patch_size=patch_size,
                                                               patch_stride=patch_stride))
        return ld_patches, hd_patches

    @staticmethod
    def images_to_patches_by_paths(paths: list[Path], minimum_scale, maximum_scale, patch_size, patch_stride):
        transform_to_tensor = transforms.ToTensor()
        image_paths = paths
        hd_patches = []
        ld_patches = []
        for image_path in image_paths:
            hd_image = Image.open(image_path).convert("YCbCr")
            # 图像降清晰处理
            scale = random.random() * (maximum_scale - minimum_scale) + minimum_scale
            ld_image = ImageUtils.scale_definition(hd_image, scale)

            # 转为tensor并分离Y层
            hd_image = transform_to_tensor(hd_image)[0].unsqueeze(0)
            ld_image = transform_to_tensor(ld_image)[0].unsqueeze(0)
            # 图像裁剪成块
            hd_patches.extend(ImageUtils.crop_image_to_patches(hd_image, patch_size=patch_size,
                                                               patch_stride=patch_stride))
            ld_patches.extend(ImageUtils.crop_image_to_patches(ld_image, patch_size=patch_size,
                                                               patch_stride=patch_stride))
        return ld_patches, hd_patches