from darea.augmentation.container import AugmentationContainerAllDarea
from darea.augmentation.container import AugmentationContainerKeywords
from darea.augmentation.container import AugmentationContainer


def get_augmentations(config, device, num_workers=0) -> AugmentationContainer:

    augmentations = config.get('augmentation', None)
    sampling_rate = config['sampling_rate']
    segment_size = config['segment_size']
    batch_size = config['batch_size']
    num_random_choose = config.get('augmentation_num_random_choose', 1)

    if augmentations is None:
        aug_train = None
        aug_val = None
    elif augmentations == 'none':
        aug_train = None
        aug_val = None
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
        aug_train = AugmentationContainerKeywords(
            sample_rate=sampling_rate,
            segment_size=segment_size,
            partition="train",
            resample=True,
            shuffle=True,
            batch_size=batch_size,
            augmentations=augmentations,
            num_random_choose=num_random_choose
        ).to(device)
        aug_val = AugmentationContainerKeywords(
            sample_rate=sampling_rate,
            segment_size=segment_size,
            partition="val",
            resample=True,
            shuffle=True,
            batch_size=batch_size,
            augmentations=augmentations,
            num_random_choose=num_random_choose
        ).to(device)

    return aug_train, aug_val
