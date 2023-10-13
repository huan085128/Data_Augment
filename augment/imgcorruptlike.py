import warnings
import cv2
import numpy as np
from utils import batch_process_images, write_results
from utils import temporary_numpy_seed

_MISSING_PACKAGE_ERROR_MSG = (
    "Could not import package `imagecorruptions`. This is an optional "
    "dependency of imgaug and must be installed manually in order "
    "to use augmenters from `imgaug.augmenters.imgcorrupt`. "
    "Use e.g. `pip install imagecorruptions` to install it. See also "
    "https://github.com/bethgelab/imagecorruptions for the repository "
    "of the package."
)

def _clipped_zoom_no_scipy_warning(img, zoom_factor):
    from scipy.ndimage import zoom as scizoom

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", ".*output shape of zoom.*")

        # clipping along the width dimension:
        ch0 = int(np.ceil(img.shape[0] / float(zoom_factor)))
        top0 = (img.shape[0] - ch0) // 2

        # clipping along the height dimension:
        ch1 = int(np.ceil(img.shape[1] / float(zoom_factor)))
        top1 = (img.shape[1] - ch1) // 2

        img = scizoom(img[top0:top0 + ch0, top1:top1 + ch1],
                      (zoom_factor, zoom_factor, 1), order=1)

        return img

def _call_imgcorrupt_func(fname, seed, convert_to_pil, *args, **kwargs):
   
    try:
        # imagecorruptions sets its own warnings filter rule via
        # warnings.simplefilter(). That rule is the in effect for the whole
        # program and not just the module. So to prevent that here
        # we use catch_warnings(), which uintuitively does not by default
        # catch warnings but saves and restores the warnings filter settings.
        with warnings.catch_warnings():
            import imagecorruptions.corruptions as corruptions
    except ImportError:
        raise ImportError(_MISSING_PACKAGE_ERROR_MSG)

    # Monkeypatch clip_zoom() as that causes warnings in some scipy versions,
    # and the implementation here suppresses these warnings. They suppress
    # all UserWarnings on a module level instead, which seems very exhaustive.
    corruptions.clipped_zoom = _clipped_zoom_no_scipy_warning
   
    image = args[0]

    input_shape = image.shape

    height, width = input_shape[0:2]
    assert height >= 32 and width >= 32, (
        "Expected the provided image to have a width and height of at least "
        "32 pixels, as that is the lower limit that the wrapped "
        "imagecorruptions functions use. Got shape %s." % (image.shape,))

    ndim = image.ndim
    assert ndim == 2 or (ndim == 3 and (image.shape[2] in [1, 3])), (
        "Expected input image to have shape (height, width) or "
        "(height, width, 1) or (height, width, 3). Got shape %s." % (
            image.shape,))

    if ndim == 2:
        image = image[..., np.newaxis]
    if image.shape[-1] == 1:
        image = np.tile(image, (1, 1, 3))

    if convert_to_pil:
        import PIL.Image
        image = PIL.Image.fromarray(image)

    with temporary_numpy_seed(seed):
        if callable(fname):
            image_aug = fname(image, *args[1:], **kwargs)
        else:
            image_aug = getattr(corruptions, fname)(image, *args[1:], **kwargs)

    if convert_to_pil:
        image_aug = np.asarray(image_aug)

    if ndim == 2:
        image_aug = image_aug[:, :, 0]
    elif input_shape[-1] == 1:
        image_aug = image_aug[:, :, 0:1]

    # this cast is done at the end of imagecorruptions.__init__.corrupt()
    image_aug = np.uint8(image_aug)

    return image_aug


"""Add fog to an image."""
def apply_fog_augmentation(image, severity=1, seed=None):
    image_np = np.array(image)
    has_alpha = image_np.shape[2] == 4
    if has_alpha:
        bgr_image, alpha_channel = image_np[:, :, :3], image_np[:, :, 3]
    else:
        bgr_image = image_np

    image_trans = _call_imgcorrupt_func("fog", seed, False, bgr_image, severity)
    
    if has_alpha:
        image_trans = np.concatenate([image_trans, alpha_channel[..., None]], axis=-1)

    return image_trans

def apply_fog(image, input_annotation, severity=1, seed=None):
    image_np = np.array(image)
    has_alpha = image_np.shape[2] == 4
    if has_alpha:
        bgr_image, alpha_channel = image_np[:, :, :3], image_np[:, :, 3]
    else:
        bgr_image = image_np

    image_trans = _call_imgcorrupt_func("fog", seed, False, bgr_image, severity)
    
    if has_alpha:
        image_trans = np.concatenate([image_trans, alpha_channel[..., None]], axis=-1)

    return image_trans, input_annotation
"""Add fog to an image."""


"""Add frost to an image."""
def apply_frost_augmentation(image, severity=1, seed=None):
    
    image_np = np.array(image)
    has_alpha = image_np.shape[2] == 4
    if has_alpha:
        bgr_image, alpha_channel = image_np[:, :, :3], image_np[:, :, 3]
    else:
        bgr_image = image_np
    image_trans = _call_imgcorrupt_func(
        "frost", seed, False, bgr_image, severity)

    if has_alpha:
        image_trans = np.concatenate(
            [image_trans, alpha_channel[..., None]], axis=-1)
    
    return image_trans


def apply_frost(image, input_annotation, severity=1, seed=None):

    image_np = np.array(image)
    has_alpha = image_np.shape[2] == 4
    if has_alpha:
        bgr_image, alpha_channel = image_np[:, :, :3], image_np[:, :, 3]
    else:
        bgr_image = image_np
    image_trans = _call_imgcorrupt_func(
        "frost", seed, False, bgr_image, severity)

    if has_alpha:
        image_trans = np.concatenate(
            [image_trans, alpha_channel[..., None]], axis=-1)

    return image_trans, input_annotation
"""Add frost to an image."""


"""Add snow to an image."""
def apply_snow_augmentation(image, severity=1, seed=None):
    
    image_np = np.array(image)
    has_alpha = image_np.shape[2] == 4
    if has_alpha:
        bgr_image, alpha_channel = image_np[:, :, :3], image_np[:, :, 3]
    else:
        bgr_image = image_np

    image_trans = _call_imgcorrupt_func("snow", seed, False, bgr_image, severity)

    if has_alpha:
        image_trans = np.concatenate(
            [image_trans, alpha_channel[..., None]], axis=-1)
        
    return image_trans

def apply_snow(image, input_annotation, severity=1, seed=None):
    
    image_np = np.array(image)
    has_alpha = image_np.shape[2] == 4
    if has_alpha:
        bgr_image, alpha_channel = image_np[:, :, :3], image_np[:, :, 3]
    else:
        bgr_image = image_np

    image_trans = _call_imgcorrupt_func("snow", seed, False, bgr_image, severity)

    if has_alpha:
        image_trans = np.concatenate(
            [image_trans, alpha_channel[..., None]], axis=-1)
        
    return image_trans, input_annotation
"""Add snow to an image."""


"""Add spatter to an image."""
def apply_spatter_augmentation(image, severity=1, seed=None):
    image_np = np.array(image)
    has_alpha = image_np.shape[2] == 4
    if has_alpha:
        bgr_image, alpha_channel = image_np[:, :, :3], image_np[:, :, 3]
    else:
        bgr_image = image_np

    image_trans = _call_imgcorrupt_func("spatter", seed, True, bgr_image, severity)

    if has_alpha:
        image_trans = np.concatenate(
            [image_trans, alpha_channel[..., None]], axis=-1)

    return image_trans

def apply_spatter(image, input_annotation, severity=1, seed=None):
    image_np = np.array(image)
    has_alpha = image_np.shape[2] == 4
    if has_alpha:
        bgr_image, alpha_channel = image_np[:, :, :3], image_np[:, :, 3]
    else:
        bgr_image = image_np

    image_trans = _call_imgcorrupt_func(
        "spatter", seed, True, bgr_image, severity)

    if has_alpha:
        image_trans = np.concatenate(
            [image_trans, alpha_channel[..., None]], axis=-1)

    return image_trans, input_annotation
"""Add spatter to an image."""


"""Add contrast to an image."""
def apply_contrast_augmentation(image, severity=1, seed=None):
    
    image_np = np.array(image)
    has_alpha = image_np.shape[2] == 4
    if has_alpha:
        bgr_image, alpha_channel = image_np[:, :, :3], image_np[:, :, 3]
    else:
        bgr_image = image_np

    image_trans = _call_imgcorrupt_func("contrast", seed, False, bgr_image, severity)

    if has_alpha:
        image_trans = np.concatenate(
            [image_trans, alpha_channel[..., None]], axis=-1)
    
    return image_trans

def apply_contrast(image, input_annotation, severity=1, seed=None):

    image_np = np.array(image)
    has_alpha = image_np.shape[2] == 4
    if has_alpha:
        bgr_image, alpha_channel = image_np[:, :, :3], image_np[:, :, 3]
    else:
        bgr_image = image_np

    image_trans = _call_imgcorrupt_func(
        "contrast", seed, False, bgr_image, severity)

    if has_alpha:
        image_trans = np.concatenate(
            [image_trans, alpha_channel[..., None]], axis=-1)

    return image_trans, input_annotation
"""Add contrast to an image."""


"""Add brightness to an image."""
def apply_brightness_augmentation(image, severity=1, seed=None):
    
    image_np = np.array(image)
    has_alpha = image_np.shape[2] == 4
    if has_alpha:
        bgr_image, alpha_channel = image_np[:, :, :3], image_np[:, :, 3]
    else:
        bgr_image = image_np
    image_trans = _call_imgcorrupt_func("brightness", seed, False, bgr_image, severity)

    if has_alpha:
        image_trans = np.concatenate(
            [image_trans, alpha_channel[..., None]], axis=-1)
        
    return image_trans

def apply_brightness(image, input_annotation, severity=1, seed=None):
    
    image_np = np.array(image)
    has_alpha = image_np.shape[2] == 4
    if has_alpha:
        bgr_image, alpha_channel = image_np[:, :, :3], image_np[:, :, 3]
    else:
        bgr_image = image_np

    image_trans = _call_imgcorrupt_func("brightness", seed, False, bgr_image, severity)

    if has_alpha:
        image_trans = np.concatenate(
            [image_trans, alpha_channel[..., None]], axis=-1)
        
    return image_trans, input_annotation
"""Add brightness to an image."""


"""Add saturate to an image."""
def apply_saturate_augmentation(image, severity=1, seed=None):
    
    image_np = np.array(image)
    has_alpha = image_np.shape[2] == 4
    if has_alpha:
        bgr_image, alpha_channel = image_np[:, :, :3], image_np[:, :, 3]
    else:
        bgr_image = image_np

    image_trans = _call_imgcorrupt_func("saturate", seed, False, bgr_image, severity)

    if has_alpha:
        image_trans = np.concatenate(
            [image_trans, alpha_channel[..., None]], axis=-1)
        
    return image_trans

def apply_saturate(image, input_annotation, severity=1, seed=None):

    image_np = np.array(image)
    has_alpha = image_np.shape[2] == 4
    if has_alpha:
        bgr_image, alpha_channel = image_np[:, :, :3], image_np[:, :, 3]
    else:
        bgr_image = image_np

    image_trans = _call_imgcorrupt_func(
        "saturate", seed, False, bgr_image, severity)

    if has_alpha:
        image_trans = np.concatenate(
            [image_trans, alpha_channel[..., None]], axis=-1)

    return image_trans, input_annotation
"""Add saturate to an image."""


"""Add jpeg compression to an image."""
def apply_jpeg_compression_augmentation(image, severity=1, seed=None):

    image_np = np.array(image)
    has_alpha = image_np.shape[2] == 4
    if has_alpha:
        bgr_image, alpha_channel = image_np[:, :, :3], image_np[:, :, 3]
    else:
        bgr_image = image_np

    image_trans = _call_imgcorrupt_func("jpeg_compression", seed, True, bgr_image, severity)

    if has_alpha:
        image_trans = np.concatenate(
            [image_trans, alpha_channel[..., None]], axis=-1)
        
    return image_trans

def apply_jpeg_compression(image, input_annotation, severity=1, seed=None):

    image_np = np.array(image)
    has_alpha = image_np.shape[2] == 4
    if has_alpha:
        bgr_image, alpha_channel = image_np[:, :, :3], image_np[:, :, 3]
    else:
        bgr_image = image_np

    image_trans = _call_imgcorrupt_func("jpeg_compression", seed, True, bgr_image, severity)

    if has_alpha:
        image_trans = np.concatenate(
            [image_trans, alpha_channel[..., None]], axis=-1)
        
    return image_trans, input_annotation
"""Add jpeg compression to an image."""


"""Add pixelate to an image."""
def apply_pixelate_augmentation(image, severity=1, seed=None):

    image_np = np.array(image)
    has_alpha = image_np.shape[2] == 4
    if has_alpha:
        bgr_image, alpha_channel = image_np[:, :, :3], image_np[:, :, 3]
    else:
        bgr_image = image_np

    image_trans = _call_imgcorrupt_func("pixelate", seed, True, bgr_image, severity)

    if has_alpha:
        image_trans = np.concatenate(
            [image_trans, alpha_channel[..., None]], axis=-1)
        
    return image_trans

def apply_pixelate(image, input_annotation, severity=1, seed=None):

    image_np = np.array(image)
    has_alpha = image_np.shape[2] == 4
    if has_alpha:
        bgr_image, alpha_channel = image_np[:, :, :3], image_np[:, :, 3]
    else:
        bgr_image = image_np

    image_trans = _call_imgcorrupt_func(
        "pixelate", seed, True, bgr_image, severity)

    if has_alpha:
        image_trans = np.concatenate(
            [image_trans, alpha_channel[..., None]], axis=-1)

    return image_trans, input_annotation
"""Add pixelate to an image."""


"""Add elastic transform to an image."""
def apply_elastic_transform_augmentation(image, severity=1, seed=None):
    
    image_np = np.array(image)
    has_alpha = image_np.shape[2] == 4
    if has_alpha:
        bgr_image, alpha_channel = image_np[:, :, :3], image_np[:, :, 3]
    else:
        bgr_image = image_np

    image_trans = _call_imgcorrupt_func("elastic_transform", seed, False, bgr_image, severity)

    if has_alpha:
        image_trans = np.concatenate(
            [image_trans, alpha_channel[..., None]], axis=-1)
        
    return image_trans

def apply_elastic_transform(image, input_annotation, severity=1, seed=None):
    
    image_np = np.array(image)
    has_alpha = image_np.shape[2] == 4
    if has_alpha:
        bgr_image, alpha_channel = image_np[:, :, :3], image_np[:, :, 3]
    else:
        bgr_image = image_np

    image_trans = _call_imgcorrupt_func("elastic_transform", seed, False, bgr_image, severity)

    if has_alpha:
        image_trans = np.concatenate(
            [image_trans, alpha_channel[..., None]], axis=-1)
        
    return image_trans, input_annotation
"""Add elastic transform to an image."""
