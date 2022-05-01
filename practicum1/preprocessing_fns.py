
import skimage
import skimage.color
import numpy as np

from tensorflow.keras.applications.mobilenet import preprocess_input
from image_processing import resize_image

def preprocessing_fn_mobilenet(image):
    image = resize_image(image, height=96, width=48)
    is_grayscale = len(image.shape) == 2

    if is_grayscale:
        color = skimage.color.gray2rgb(image)
    else:
        assert image.shape[2] == 3
        # https://github.com/keras-team/keras/blob/3a33d53ea4aca312c5ad650b4883d9bac608a32e/keras/applications/imagenet_utils.py#L199
        is_convert_bgr_to_rgb = True
        if is_convert_bgr_to_rgb:
            color = image[:,:, ::-1]
        else:
            color = image
    img_data = np.expand_dims(color, axis=0)
    processed = preprocess_input(img_data)
    return processed

