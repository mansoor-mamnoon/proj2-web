import numpy as np
import cv2
from scipy.signal import convolve2d

def box_filter(size = 9):

    """
    Creates a box filter of size n x n filled with ones scaled down so that the sum of its entries is 1
    
    """

    assert size % 2 == 1 and size > 0, "size must a positive odd number"

    the_box = np.ones((size, size),  dtype = np.float32) # create an nxn array of ones filled with floating points

    the_box_sized_down = the_box / (size * size) # make sure the sum of all the entries in the filter is 1 need regular division not round down integer division because then every image intensity becomes 0 

    return the_box_sized_down

def x_diff():

    """

    creates a 1 x 2 kernel to measure change in the x direction of the image
    
    
    """

    kernel_horizontal = np.array([[1.0, 0, -1.0]], dtype = np.float32) # cast as float to make sure that kernel dtype and image dtype match

    return kernel_horizontal

def y_diff():

    """

    creates a 2 x 1 kernel to measure change in the y direction of the image
    
    
    """

    kernel_vertical = np.array([[1.0],
                                [0.0], 
                                [-1.0]], dtype = np.float32)

    return kernel_vertical


def gaussian_kernel(size, sigma):

    """

    returns a gaussian kernel of size size x size with standard deviation sigma
    
    sum of weights in kernel returned is 1

    """

    assert size % 2 == 1, "enter an odd size" # I want a well-defined center pixel to be applying my kernel to 
    
    assert size > 0 , "enter a positive size" # to make sure someone does not enter negative sizes

    gaussian1D = cv2.getGaussianKernel(ksize = size, sigma = sigma) # returns a 1 x size array

    gaussian2D = (gaussian1D @ gaussian1D.T).astype(np.float32) # this returns a size by size array

    sum_check = gaussian2D.sum() # make sure sum of entries is 1
    
    if sum_check != 1:

        gaussian2D = gaussian2D/sum_check

    return gaussian2D


def derivative_of_gauss_kernel(size, sigma):
    
    """

    creates two derivative of gaussian kernels: 1 in x direction and 1 in y direction

    gaussian used is size x size and has standard deviation sigma

    """

    from filters import x_diff, y_diff # fix import issue

    gauss = gaussian_kernel(size, sigma)

    x_derivative_kernel = x_diff()

    y_derivative_kernel = y_diff()

    # I return images of same size 0 padded

    kx = convolve2d(gauss, x_derivative_kernel, mode='full', boundary='fill', fillvalue=0).astype(np.float32)
    ky = convolve2d(gauss, y_derivative_kernel, mode='full', boundary='fill', fillvalue=0).astype(np.float32)

    return kx.astype(np.float32), ky.astype(np.float32)

def zero_to_one_normalizer(arr):

    """
    Normalizes all arrays to have values between 0 and 1

    uses the formula that is applied to each element in the array:

    new_value = (old_value - min) / (max - min)
    
    
    """

    min_value, max_value = np.min(arr), np.max(arr)

    return (arr - min_value) / (max_value - min_value)
















