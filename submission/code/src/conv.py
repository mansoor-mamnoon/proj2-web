import numpy as np


def pad_zero(image, pad_height, pad_width):

    """

    takes an image and adds zeros around it with each border of image being
    of width pad_width greater and height pad_height greater

    returns the image padded

    """

    original_h, original_w = image.shape

    padded_image = np.zeros((original_h + 2 * pad_height, original_w + 2 * pad_width), dtype = np.float32) # this creates base for me to put image into. black border around image

    padded_image[pad_height: original_h + pad_height, pad_width: original_w + pad_width] = image # centres the image inside the black base created

    return padded_image


def flip_kernel(kernel):

    """

    flips kernel in both horizontal and in vertical axes
    
    
    """

    return kernel[::-1, ::-1] # the third argument is orientation of kernel in that axis. I want the whole slice of kernel but in reverse orientation

def conv_4_loops(image, kernel):

    """
    Applies the kernel to the image in 4 loops using brute force multiplication
    
    """

    kernel_height, kernel_width = kernel.shape

    image_height_original, image_width_original = image.shape

    padding_height, padding_width = kernel_height//2, kernel_width// 2 # this keeps the dimensions of the output image the same

    kernel_flipped = flip_kernel(kernel)

    image_padded = pad_zero(image, padding_height, padding_width)

    output_image = np.zeros_like(image, dtype = np.float32) # instead of np.zeros. This creates an array of shape image without having to specify the dimensions

    for i in range(image_height_original):
        for j in range(image_width_original):

            new_intensity = 0

            for a in range(kernel_height):

                for b in range(kernel_width):

                    new_intensity = new_intensity + image_padded[i + a, j + b] * kernel_flipped[a, b] # 

            output_image[i, j] = new_intensity

    return output_image

def conv_2_loops(image, kernel):

    """

    applies the kernel to the image in 2 loops instead of 4
    
    
    """

    kernel_height, kernel_width = kernel.shape
    image_height, image_width = image.shape

    padded_height, padded_width = kernel_height // 2, kernel_width // 2

    kernel_flipped = flip_kernel(kernel)

    padded_image = pad_zero(image, padded_height, padded_width)

    output_image = np.zeros_like(image, dtype = np.float32)

    for i in range(image_height):

        for j in range(image_width):

            image_window_to_inspect = padded_image[i: i + kernel_height, j: j + kernel_width]
            output_image[i, j] = np.sum(image_window_to_inspect * kernel_flipped, dtype= np.float32) # numpy backend handles elementwise multiplication and then i add all the multiplications together


    return output_image




                



















