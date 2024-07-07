import random
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from utils import ImageUtils


class SRCNNDataset(Dataset):
    def __init__(self, paths, minimum_scale, maximum_scale, patch_size, patch_stride):
        self.transform_to_tensor = transforms.ToTensor()
        self.paths = paths
        self.minimum_scale = minimum_scale
        self.maximum_scale = maximum_scale
        self.patch_size = patch_size
        self.patch_stride = patch_stride

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        hd_image = Image.open(self.paths[idx]).convert("YCbCr")
        # 图像降清晰处理
        scale = random.random() * (self.maximum_scale - self.minimum_scale) + self.minimum_scale
        ld_image = ImageUtils.scale_definition(hd_image, scale)
        # 转为tensor并分离Y层
        hd_image = self.transform_to_tensor(hd_image)[0].unsqueeze(0)
        ld_image = self.transform_to_tensor(ld_image)[0].unsqueeze(0)
        ld_patches = ImageUtils.crop_image_to_patches(ld_image, patch_size=self.patch_size,
                                                      patch_stride=self.patch_stride)
        hd_patches = ImageUtils.crop_image_to_patches(hd_image, patch_size=self.patch_size,
                                                      patch_stride=self.patch_stride)
        return ld_patches, hd_patches


class SRCNNDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs
        super().__init__(*args, **kwargs)
        self.collate_fn = self.custom_collate_fn

    def custom_collate_fn(self, bath):
        total_ld_patches = []
        total_hd_patches = []
        for ld_patches, hd_patches in bath:
            total_ld_patches.extend(ld_patches)
            total_hd_patches.extend(hd_patches)
            # 将图像块打乱
        combined = list(zip(total_ld_patches, total_hd_patches))
        if self.kwargs['shuffle']:
            random.shuffle(combined)
        return combined
