import numpy as np
from scipy.signal import convolve2d
from filters import gaussian_kernel, zero_to_one_normalizer
from freq import is_color, apply_func_to_each_channel, image_matcher


def gaussian_stack(image, levels, size = 9, sigma = 2.0):

    """

    builds a list of images with no downsampling each image

    having a gaussian blur applied to the image before it and returns the list of images
    
    """

    stack_of_images = [image] # store all images in this array

    gaussian = gaussian_kernel(size, sigma)

    for level in range(1, levels): 

        if is_color(stack_of_images[-1]): # each channel is blurred independently with the same kernel

            next_level = apply_func_to_each_channel(stack_of_images[-1], lambda channel: convolve2d(channel, gaussian, mode = "same", boundary = "fill", fillvalue = 0))

        else:

            next_level = convolve2d(stack_of_images[-1], gaussian, mode = "same", boundary = "fill", fillvalue = 0)

        stack_of_images.append(next_level) # append adds to end of the array

    return stack_of_images

def laplacian_stack(image, levels, size = 9, sigma = 2.0):

    """

    first creates gaussian stack

    then finds the difference between consecutive members of gaussian stack to form each component

    of the laplacian stack
    
    returns array of laplacian matrices

    """

    gaussians = gaussian_stack(image, levels, size, sigma)

    laplacian_stack_images = [] # holds laplacian matrices

    for i in range(levels - 1): # there is one fewer laplacian than gaussian

        laplacian_stack_images.append(gaussians[i] - gaussians[i + 1]) # consecutive gaussians subtracted

    laplacian_stack_images.append(gaussians[-1]) # remaining low frequency images captured by last gaussian

    return laplacian_stack_images

def tile_stacked_sideways(stack_of_images):

    """

    takes in stack of images and concatenates them side by side for easy display and debugging

    and returns images concatenated side to side
    
    
    """

    tiles = [] # I store all the tiles in this array

    for level in stack_of_images:

        tiles.append(zero_to_one_normalizer(level))
    
    return np.concatenate(tiles, axis = 1)

def mask_horiz(image_height, image_width, ramp=100):  # horizontal mask: top=1, bottom=0
    # I just reused the vertical mask and transpose it

    mask = make_soft_vertical_mask(image_width, image_height).T

    return mask

def mask_circle(image_height, image_width, radius=None, ramp=60):  # radial mask from center
    
    if radius is None:

        radius = min(image_height, image_width) * 0.35

    y_indices = np.arange(image_height).reshape(-1, 1)

    x_indices = np.arange(image_width).reshape(1, -1)

    center_y, center_x = image_height // 2, image_width // 2

    distance = np.sqrt((y_indices - center_y) ** 2 + (x_indices - center_x) ** 2)

    mask = (radius + ramp - distance) / ramp

    mask = np.clip(mask, 0.0, 1.0)

    return mask

def mask_diag(image_height, image_width, ramp=80):  # diagonal mask TL→BR

    y_indices = np.arange(image_height).reshape(-1, 1)

    x_indices = np.arange(image_width).reshape(1, -1)

    values = x_indices - y_indices + (image_width - image_height) // 2

    mask = values / float(ramp)

    mask = 0.5 + 0.5 * np.tanh(mask)

    mask = np.clip(mask, 0.0, 1.0)
    
    return mask

def mask_horiz(image_height, image_width, ramp=100):  # horizontal mask: 1 at top and 0 at bottom
    # I transpose a vertical (left→right) cosine mask to become top→bottom, and pass ramp through
    mask = make_soft_vertical_mask(image_width, image_height, ramp=ramp).T 
    return mask



def multiband_blend(image1, image2, mask, levels=5, size=9, sigma=2.0):

    """
    multi-resolution blend using Gaussian/Laplacian stacks (no downsampling)
    
    
    Lout[i] = GM[i] * LA[i] + (1 - GM[i]) * LB[i]

    final image = sum_i Lout[i], then clip to [0,1]

    """

    image1rescaled, image2rescaled = image_matcher(image1, image2) # make sure images 1 and 2 have same dimensions

    if mask.ndim == 2 and is_color(image1rescaled):
        mask_rescaled = np.repeat(mask[..., None], 3, axis=2)
    else:
        mask_rescaled = mask

    
    mask_rescaled, _ = image_matcher(mask_rescaled, image1rescaled) # make mask same size as image it is masking
        
    
    G_image1 = gaussian_stack(image1rescaled, levels, size=size, sigma=sigma) # stacks of each type of each rescaled image are created
    G_image2 = gaussian_stack(image2rescaled, levels, size=size, sigma=sigma)
    L_image1 = laplacian_stack(image1rescaled, levels, size=size, sigma=sigma)
    L_image2 = laplacian_stack(image2rescaled, levels, size=size, sigma=sigma)
    G_mask = gaussian_stack(mask_rescaled, levels, size=size, sigma=sigma)


    output_image_stack = []
    for i in range(levels):
        output_image_stack.append(G_mask[i] * L_image1[i] + (1.0 - G_mask[i]) * L_image2[i]) # apply formula in paper here


    output_image = np.zeros_like(image1rescaled, dtype=np.float32) # sum each each image in the stack in this loop
    for i in range(levels):
        output_image = output_image + output_image_stack[i].astype(np.float32) 

    output_image = np.clip(output_image, 0.0, 1.0) # here I make sure values always fall between 0 and 1

    return output_image, G_image1, G_image2, G_mask, L_image1, L_image2, output_image_stack   # returning internals for use in other functions


def make_soft_vertical_mask(image_height, image_width, ramp=100):

    """

    use the cosine function to weight values within ramp distance of center of image in their

    contribution to the merged image
    
    """


    canvas = np.zeros((image_height, image_width), dtype=np.float32)

    center = image_width // 2

    left  = max(0, center - ramp)

    right = min(image_width, center + ramp)

    canvas[:, :left] = 1.0 # full effect on this side of image

    if right > left: # cosine ramp from 1 to 0 across [left, right)
        x = np.linspace(0, np.pi, right - left, endpoint=False, dtype=np.float32) # created a uniform space of cosine values shifted such that range is from 0 to 1 within ramp space
        
        canvas[:, left:right] = 0.5 * (1.0 + np.cos(x))
    
    canvas[:, right:] = 0.0 # zero effect on this side of image
    
    return canvas
    
    















