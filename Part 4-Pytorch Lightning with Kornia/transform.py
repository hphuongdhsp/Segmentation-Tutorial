import albumentations as A
import kornia as K
from torch import Tensor


def train_transform(p: float = 0.5):
    return A.Compose(
        [
            A.HorizontalFlip(),
            A.ShiftScaleRotate(
                shift_limit=0,
                scale_limit=(-0.2, 0.2),
                rotate_limit=(-30, 30),
                interpolation=1,
                border_mode=0,
                value=(0, 0, 0),
                p=0.5,
            ),
            A.GaussNoise(var_limit=(0, 50.0), mean=0, p=0.4),
            A.GaussianBlur(blur_limit=(3, 7), p=0.4),
            A.OneOf(
                [
                    A.HueSaturationValue(
                        hue_shift_limit=10,
                        sat_shift_limit=15,
                        val_shift_limit=15,
                        p=0.5,
                    ),
                    A.CLAHE(clip_limit=2),
                    A.RandomBrightnessContrast(
                        brightness_limit=0.15,
                        contrast_limit=0.3,
                        brightness_by_max=True,
                        p=0.5,
                    ),
                ],
                p=0.3,
            ),
            A.Resize(384, 384, always_apply=True),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255.0,
                always_apply=False,
                p=1.0,
            ),
        ],
        p=p,
    )


def valid_transform(p: float = 1):
    return A.Compose(
        [
            A.Resize(384, 384, always_apply=True),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255.0,
                always_apply=False,
                p=1.0,
            ),
        ],
        p=p,
    )


def resize(p: float = 1):
    return A.Resize(384, 384, always_apply=True)


def normalize(p: float = 1):
    return A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        max_pixel_value=255.0,
        always_apply=False,
        p=1.0,
    )


valid_transform_K = K.augmentation.AugmentationSequential(
    K.augmentation.Normalize(Tensor((0.485, 0.456, 0.406)), Tensor((0.229, 0.224, 0.225)), p=1),
    data_keys=["input", "mask"],
)

train_transform_K = K.augmentation.AugmentationSequential(
    K.augmentation.container.ImageSequential(  # OneOf
        K.augmentation.RandomHorizontalFlip(p=0.6),
        K.augmentation.RandomVerticalFlip(p=0.6),
        random_apply=1,
        random_apply_weights=[0.5, 0.5],
    ),
    K.augmentation.ColorJitter(0.1, 0.1, 0.1, 0.1, p=0.5),
    # K.augmentation.RandomAffine( degrees = (-15.0,15.0), p= 0.3),
    K.augmentation.Normalize(Tensor((0.485, 0.456, 0.406)), Tensor((0.229, 0.224, 0.225)), p=1),
    data_keys=["input", "mask"],
    same_on_batch=False,
)
