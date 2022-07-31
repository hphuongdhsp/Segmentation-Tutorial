import albumentations as A


def valid_transform():
    return A.Compose(
        [
            A.Resize(384, 384, always_apply=True),
        ],
        p=1,
    )


def train_transform():
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
        ],
        p=1,
    )
