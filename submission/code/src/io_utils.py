import cv2
import numpy as np
from pathlib import Path

def to_float(arr):
    """
    converts all image arrays to form I plan on working with, which is a array
    of floating points

    input: array representing an image
    output: image array converted to floating point

    """

    arr = arr.astype(np.float32)

    if arr.max() > 1.0: # if array is in [0, 255] format
        arr = arr / 255.0
    
    return np.clip(arr, 0, 1) # to prevent overflow due to rounding errors

def load_gray(path_to_image):
    """
    Loads image stored at input specified as a grayscale image and returns the image
    
    """

    picture = Path(path_to_image) # create path object with input as way to image

    if not picture.exists(): # added because picture was not loading

        raise FileNotFoundError(f"Image not found at {path_to_image}")
    
    black_and_white = cv2.imread(str(picture), cv2.IMREAD_GRAYSCALE) # convert image to numpy array in grayscale

    if black_and_white is None:
        
        raise FileNotFoundError(f"Failed to read image at {path_to_image}")
    
    return to_float(black_and_white)


def load_color(path_to_image):

    """

    loads the image in color in the RGB format. I chose RGB to standardise
    across the project

    Works the same as load_gray function right before this
    
    
    """

    picture = Path(path_to_image) # create path object with input as way to image

    if not picture.exists(): # added because picture was not loading

        raise FileNotFoundError(f"Image not found at {path_to_image}")
    
    color_image = cv2.imread(str(picture), cv2.IMREAD_COLOR) # convert image to numpy array in grayscale

    if color_image is None:
        
        raise FileNotFoundError(f"Failed to read image at {path_to_image}")
    
    rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB) # convert bgr of opencv to rgb. Good I saw this about opencv library. Debugging would have been a pain
    
    return to_float(rgb)

def image_saver(path_to_save_to, image):

    """

    saves image to local storage

    saves depending on whether image is black and white or color
    
    """

    storage_location = Path(path_to_save_to) 

    storage_location.parent.mkdir(parents = True, exist_ok = True) # at start out folder does not exist. i create it here and also if parents specified don't exist I create that too

    image_array_cleaned_up = np.clip(image, 0 , 1) # make sure that image is a float of between 0 and 1 with no overflow

    image_array_cleaned_up = (image_array_cleaned_up * 255.0).astype(np.uint8) # to save to jpg or png I need to cast as unsigned 8 bit integer

    if image_array_cleaned_up.ndim == 2: # if black and white image

        did_it_write = cv2.imwrite(str(storage_location), image_array_cleaned_up)

    elif image_array_cleaned_up.ndim == 3 and image_array_cleaned_up.shape[2] == 3: # color image where third channel is 3 channels of color

        bgr = cv2.cvtColor(image_array_cleaned_up, cv2.COLOR_RGB2BGR) # convert to BGR format expected by cv2
        did_it_write = cv2.imwrite(str(storage_location), bgr)

    else:

        raise ValueError("image format incorrect")
    
    if not did_it_write:

        raise IOError(f"Failed to save image to {path_to_save_to}")


    





    





















