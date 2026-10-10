import random
import torch
from torch.utils.data import Dataset
from Data_generation import compute_targets
from Sensor_maps import load_rgb, rgb_to_gray, random_gaussian_blur, make_sensor_maps

def random_crop_pair(gray_hw, target_chw, crop_h, crop_w):
    gh, gw = gray_hw.shape
    _, th, tw = target_chw.shape

    if gh != th or gw != tw:
        raise ValueError(
            f"gray and target size mismatch before crop: "
            f"gray={gray_hw.shape}, target={target_chw.shape}"
        )

    if gh < crop_h or gw < crop_w:
        raise ValueError(
            f"Crop size {(crop_h, crop_w)} is larger than gray image {(gh, gw)}"
        )

    top = random.randint(0, gh - crop_h)
    left = random.randint(0, gw - crop_w)

    gray_crop = gray_hw[top:top + crop_h, left:left + crop_w]
    target_crop = target_chw[:, top:top + crop_h, left:left + crop_w]

    return gray_crop, target_crop


def center_crop_pair(gray_hw, target_chw, crop_h, crop_w):
    gh, gw = gray_hw.shape
    _, th, tw = target_chw.shape

    if gh != th or gw != tw:
        raise ValueError(
            f"gray and target size mismatch before crop: "
            f"gray={gray_hw.shape}, target={target_chw.shape}"
        )

    if gh < crop_h or gw < crop_w:
        raise ValueError(
            f"Crop size {(crop_h, crop_w)} is larger than gray image {(gh, gw)}"
        )

    top = (gh - crop_h) // 2
    left = (gw - crop_w) // 2

    gray_crop = gray_hw[top:top + crop_h, left:left + crop_w]
    target_crop = target_chw[:, top:top + crop_h, left:left + crop_w]

    return gray_crop, target_crop


class TerrainDataset(Dataset):
    def __init__(
        self,
        image_paths,
        train=True,
        crop_size=(64, 64),
        blur_probability=0.5
    ):
        self.image_paths = image_paths
        self.train = train
        self.crop_h, self.crop_w = crop_size
        self.blur_probability = blur_probability

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]

        rgb = load_rgb(image_path)              
        gray = rgb_to_gray(rgb)  

        if self.train:
            gray = random_gaussian_blur(gray, p=self.blur_probability, sigma_min=0.2, sigma_max=1.2)
        target = compute_targets(gray[None, :, :])

        if self.train:
            gray, target = random_crop_pair(gray, target, self.crop_h, self.crop_w)
        else:
            gray, target = center_crop_pair(gray, target, self.crop_h, self.crop_w)

        if not isinstance(target, torch.Tensor):
            target = torch.from_numpy(target).float()
        else:
            target = target.float()
    
        x = make_sensor_maps(gray)              # (5, H, W)
        y = target                              # (3, H, W)

        return x, y, image_path
