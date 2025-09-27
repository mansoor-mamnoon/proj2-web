import numpy as np
from scipy.signal import convolve2d
from filters import x_diff, y_diff

def gradient_image_xy(image):

    """

    creates the x gradient image and y gradient image and return both images
    
    
    """

    x_diff_kernel = x_diff()

    y_diff_kernel = y_diff()


    x_gradient_image = convolve2d(image, x_diff_kernel, mode = "same", boundary = 'fill', fillvalue = 0).astype(np.float32) # keeps processed image the same size. cast as float 32 at the end. 0 pad the image
    y_gradient_image = convolve2d(image, y_diff_kernel, mode = "same", boundary = 'fill', fillvalue = 0).astype(np.float32)

    return x_gradient_image, y_gradient_image

def gradient_magnitude_image(image, x_gradient_image, y_gradient_image, normalize_for_display):

    """

    computes the gradient magnitude of image given the x gradient image and y gradient image. 

    I add normalize_for_display flag to make sure calculations can be done on image and it can also be shown without
    having to create a separate function for either
    
    
    """

    gradient_magnitude = np.sqrt(x_gradient_image.astype(np.float32) ** 2 + y_gradient_image.astype(np.float32) ** 2) # added float32 casting here to ensure if it receives non float 32 images from elsewhere, it still works

    if normalize_for_display:

        max_value = gradient_magnitude.max()

        if max_value > 0: # division by zero not possible

            gradient_magnitude = gradient_magnitude / max_value # make sure image is float between 0 and 1

    return gradient_magnitude.astype(np.float32)

def binarize_edges(gradient_mag_image, threshold):

    """

    decides whether an edge is an edge or not by only counting an edge if its edge 
    strength is above edge strength threshold
    
    
    """

    return (gradient_mag_image > threshold).astype(np.float32) 

def signed_image_viewer(image):

    """

    displays 0 as grey, negatives as closer to black and postives as closer to white
    
    
    """
    

    image_casted = image.astype(np.float32)

    max_value = np.max(np.abs(image_casted))

    return 0.5 + 0.5 * (image_casted / max_value ) # image casted over max value gives me values between -1 and 1 . Multiplying by 0.5 gives me values between -0.5 and 0.5 . Adding 0.5 then gives me values between 0 and 1


















