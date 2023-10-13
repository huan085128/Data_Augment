import os
import cv2
import random
import numbers
import sys
import functools
import numpy as np


def is_image_or_path(input_object):
    if isinstance(input_object, str):
        if os.path.isfile(input_object):
            # Check if the file has a supported image extension
            supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            file_extension = os.path.splitext(input_object)[1].lower()

            if file_extension in supported_extensions:
                return "image_path"
            else:
                return "not_image_path"
        else:
            return "not_image_path"

    elif isinstance(input_object, (list, tuple, set)):
        if len(input_object) == 3 and all(isinstance(item, int) for item in input_object):
            return "color_tuple"

    elif isinstance(input_object, cv2.UMat) or isinstance(input_object, np.ndarray):
        return "image"

    else:
        return "unknown"

def batch_process_images(process_image_func):
    """Decorator to process a batch of images."""
    @functools.wraps(process_image_func)
    def wrapper(input_path, output_folder, *args, **kwargs):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        if os.path.isfile(input_path):
            process_image_func(input_path, output_folder, *args, **kwargs)
        elif os.path.isdir(input_path):
            for file_name in os.listdir(input_path):
                file_path = os.path.join(input_path, file_name)
                if os.path.isfile(file_path):
                    process_image_func(
                        file_path, output_folder, *args, **kwargs)
                else:
                    print(f"Skipping non-file {file_path}")
        else:
            print(f"Input path not found: {input_path}")

    return wrapper


def write_results(image, image_path, output_folder):
    """Write the results of the image processing to disk."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_path = os.path.join(output_folder, os.path.basename(image_path))
    cv2.imwrite(output_path, image)
    print(f"Processed {image_path} -> {output_path}")

def is_single_number(val):
    """Check if a value is a single number (int or float)."""
    is_single_integer = isinstance(val, numbers.Integral) and not isinstance(val, bool)
    is_single_float = isinstance(val, numbers.Real) \
                      and not is_single_integer(val) \
                      and not isinstance(val, bool)
    
    return is_single_integer or is_single_float

def is_callable(val):
    """Check if a value is callable (function, class, ...)."""
    if sys.version_info[0] == 3 and sys.version_info[1] <= 2:
        return hasattr(val, '__call__')
    return callable(val)

def _check_value_range(value, name, value_range):
    if value_range is None:
        return True

    if isinstance(value_range, tuple):
        assert len(value_range) == 2, (
            "If 'value_range' is a tuple, it must contain exactly 2 entries, "
            "got %d." % (len(value_range),))

        if value_range[0] is None and value_range[1] is None:
            return True

        if value_range[0] is None:
            assert value <= value_range[1], (
                "Parameter '%s' is outside of the expected value "
                "range (x <= %.4f)" % (name, value_range[1]))
            return True

        if value_range[1] is None:
            assert value_range[0] <= value, (
                "Parameter '%s' is outside of the expected value "
                "range (%.4f <= x)" % (name, value_range[0]))
            return True

        assert value_range[0] <= value <= value_range[1], (
            "Parameter '%s' is outside of the expected value "
            "range (%.4f <= x <= %.4f)" % (
                name, value_range[0], value_range[1]))

        return True

    if is_callable(value_range):
        value_range(value)
        return True

    raise Exception("Unexpected input for value_range, got %s." % (
        str(value_range),))


def batch_process_images_trans(process_image_func):
    """Decorator for batch processing of images and annotations."""
    @functools.wraps(process_image_func)
    def wrapper(self, input_image_folder, input_annotation_folder,
                output_image_folder, output_annotation_folder,
                *args, **kwargs):
        if not os.path.exists(output_image_folder):
            os.makedirs(output_image_folder)
        if not os.path.exists(output_annotation_folder):
            os.makedirs(output_annotation_folder)

        image_files = [f for f in os.listdir(
            input_image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

        for image_file in image_files:
            input_image_path = os.path.join(input_image_folder, image_file)
            input_annotation_path = os.path.join(
                input_annotation_folder, os.path.splitext(image_file)[0])

            output_image_path = os.path.join(output_image_folder, image_file)
            output_annotation_path = os.path.join(
                output_annotation_folder, os.path.splitext(image_file)[0])

            process_image_func(self, input_image_path, input_annotation_path,
                               output_image_path, output_annotation_path, *args, **kwargs)

    return wrapper

class temporary_numpy_seed(object):
    """Context to temporarily alter the random state of ``numpy.random``.
    The random state's internal state will be set back to the original one
    once the context finishes.
    Added in 0.4.0.
    Parameters
    ----------
    entropy : None or int
        The seed value to use.
        If `None` then the seed will not be altered and the internal state
        of ``numpy.random`` will not be reset back upon context exit (i.e.
        this context will do nothing).
    """
    # pylint complains about class name
    # pylint: disable=invalid-name

    def __init__(self, entropy=None):
        self.old_state = None
        self.entropy = entropy

    def __enter__(self):
        if self.entropy is not None:
            self.old_state = np.random.get_state()
            np.random.seed(self.entropy)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.entropy is not None:
            np.random.set_state(self.old_state)