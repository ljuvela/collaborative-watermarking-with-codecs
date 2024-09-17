from darea.augmentation.container import AugmentationContainerAllDarea
from darea.augmentation.container import AugmentationContainerKeywords
from darea.augmentation.container import AugmentationContainer


def get_augmentations(config, device, num_workers=0) -> AugmentationContainer:

    augmentations = config.get('augmentation', None)
    augmentations_val = config.get('augmentation_validation', None)
    if augmentations_val is None:
        augmentations_val = augmentations

    sampling_rate = config['sampling_rate']
    segment_size = config['segment_size']
    batch_size = config['batch_size']
    num_random_choose = config.get('augmentation_num_random_choose', 1)
    grad_clip_norm_level = config.get('grad_clip_norm_level', None)

    if augmentations is None:
        aug_train = None
    elif augmentations == 'none':
        aug_train = None
    elif augmentations == 'all':
        aug_train = AugmentationContainerAllDarea(
            sample_rate=sampling_rate,
            segment_size=segment_size,
            partition="train",
            resample=True,
            shuffle=True,
            batch_size=batch_size,
            num_random_choose=num_random_choose
        ).to(device)
    else:
        aug_train = AugmentationContainerKeywords(
            sample_rate=sampling_rate,
            segment_size=segment_size,
            partition="train",
            resample=True,
            shuffle=True,
            batch_size=batch_size,
            augmentations=augmentations,
            num_random_choose=num_random_choose,
            grad_clip_norm_level=grad_clip_norm_level
        ).to(device)

    if augmentations_val is None:
        aug_val = None
    elif augmentations_val == 'none':
        aug_val = None
    elif augmentations_val == 'all':
        aug_val = AugmentationContainerAllDarea(
            sample_rate=sampling_rate,
            segment_size=segment_size,
            partition="val",
            resample=True,
            shuffle=True,
            batch_size=batch_size,
            num_random_choose=num_random_choose
        ).to(device)
    else:
        aug_val = AugmentationContainerKeywords(
            sample_rate=sampling_rate,
            segment_size=segment_size,
            partition="val",
            resample=True,
            shuffle=True,
            batch_size=batch_size,
            augmentations=augmentations_val,
            num_random_choose=num_random_choose
        ).to(device)

    return aug_train, aug_val
