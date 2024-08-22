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



def get_augmentations_eval(config, device, batch_size=None, num_workers=0) -> AugmentationContainer:
    """ get_augmentations_eval(config, device, num_workers=0, batch_size=None)
    
    return augmentation for evaluation
    """
    augmentations = config.get('augmentation_evaluation', None)

    sampling_rate = config['sampling_rate']
    segment_size = config['segment_size']

    # we need to specify batch size based on configuration of boostrp reps
    if batch_size is None:
        batch_size = config['batch_size']
        
    num_random_choose = config.get('augmentation_num_random_choose', 1)

    if augmentations is None:
        aug_eval = None
        aug_name = ['None']
    elif augmentations == 'none':
        aug_eval = None
        aug_name = ['None']        
    elif augmentations == 'all':
        aug_eval = AugmentationContainerAllDarea(
            sample_rate=sampling_rate,
            segment_size=segment_size,
            partition="train",
            resample=True,
            shuffle=True,
            batch_size=batch_size,
            num_random_choose=num_random_choose
        ).to(device)
        # this should be returned by AugmentationContainerAllDarea
        aug_name = ["noise", "reverb", "codec_mp3_32kbit", "codec_ogg_vorbis_32kbit"]
    else:
        aug_eval = AugmentationContainerKeywords(
            sample_rate=sampling_rate,
            segment_size=segment_size,
            partition="train",
            resample=True,
            shuffle=True,
            batch_size=batch_size,
            augmentations=augmentations,
            num_random_choose=num_random_choose
        ).to(device)
        aug_name = augmentations

    return aug_eval, aug_name
