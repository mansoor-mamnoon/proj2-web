import numpy as np
from scipy.signal import convolve2d

from filters import gaussian_kernel

def to_gray_1ch(inoutimage):
    """
    convert HxWx3 color to single-channel grayscale by simple average.
    if already HxW, just return it.
    
    """
    if inoutimage.ndim == 2:
        return inoutimage
    return inoutimage.mean(axis=2)

def is_color(image):

    """

    tells me whether an image is a color image by looking at format of pixel
    arrangement
    
    """

    return (image.ndim == 3) and (image.shape[2] == 3)

def apply_func_to_each_channel(image, func):

    """

    apply function to each channel of image separately and return the new image created
    
    because scipy has functions that only work on 2D arrays
    
    """

    output_image = np.empty_like(image, dtype = np.float32) # faster than zeros also I overwrite all the values anyways

    for channel in range(3):

        output_image[..., channel] = func(image[..., channel]).astype(np.float32)

    return output_image


def gray_blur(image, size, sigma):

    """

    blurs image using gaussian blur with standard deviation set to sigma
    and gaussian kernel of size size x size
    
    
    """

    gauss = gaussian_kernel(size, sigma)

    return convolve2d(image, gauss, mode="same", boundary="fill", fillvalue=0).astype(np.float32)


def image_shapener(image, size, sigma, amount):

    """

    sharpens the image by using this formula

    low = gaussian blur of image

    high = image - low

    sharpened_image = image + sharpening_factor * high
    
    """

    image_cast = image.astype(np.float32)

    if is_color(image):
        low_freq = apply_func_to_each_channel(image_cast, lambda channel: gray_blur(channel, size, sigma))

    else:

        low_freq = gray_blur(image_cast, size = size, sigma = sigma)

    high_freq = image_cast - low_freq

    sharp_image = np.clip(image_cast + high_freq * amount, 0, 1)

    return low_freq.astype(np.float32), high_freq.astype(np.float32), sharp_image.astype(np.float32)

def low_pass_filter(image, size, sigma):

    """

    returns gaussian filter of image with gaussian paramaters of standard deviation
    sigma and using a blur matrix of size size x size
    
    
    """

    image_cast = image.astype(np.float32)

    if is_color(image_cast):

        low_freq = apply_func_to_each_channel(image_cast, lambda channel: gray_blur(channel, size, sigma))

    else:

        low_freq = gray_blur(image_cast, size = size, sigma = sigma)
    
    return low_freq

def high_pass_filter(image, size, sigma):

    """
    
    returns high frequency part of image, which is image - blur
    
    """

    return image.astype(np.float32) - low_pass_filter(image, size, sigma).astype(np.float32)

def translator(image, x_direction, y_direction):

    """
    translates the image in the x direction where positive is right side 

    y direction where positive is down

    no roll around occurs unlike np.roll
    
    """

    image_cast = image.astype(np.float32)

    image_height, image_width = image_cast.shape[:2] # height and width of image

    output_image = np.zeros_like(image_cast, dtype = np.float32) #canvas to put the image into initially filled with 0s

    old_row_start = max(0, -y_direction) # suppose image is shifted down by 5 (y_direction = 5) then the top of the image is still valid.

    old_row_end = min(image_height, image_height - y_direction) # then the bottom of the image would have run out of the page

    old_column_start = max(0, -x_direction)

    old_column_end = min(image_width, image_width - x_direction)

    new_row_start = max(0, y_direction) # the image will be copied 5 spaces below its original position

    new_row_end = min(image_height, image_height + y_direction)

    new_column_start = max(0, x_direction)

    new_column_end = min(image_width, image_width + x_direction)

    if old_row_start < old_row_end and old_column_start < old_column_end: # this ensures that some part of image still exists after shift. If both are out of bounds because shift values were too large nothing will be returned 

        if is_color(image_cast): # in this case return channels as it is. In B/W case no channels to be returned

            output_image[new_row_start:new_row_end, new_column_start:new_column_end, : ] = image_cast[old_row_start: old_row_end, old_column_start: old_column_end, : ]

        else:

            output_image[new_row_start:new_row_end, new_column_start:new_column_end] = image_cast[old_row_start: old_row_end, old_column_start: old_column_end]

    return output_image


def image_matcher(image1, image2):

    """

    matches images 1 and 2 by cropping edges off equally until they both have the

    dimensions. Then it returns both cropped images
    
    
    """

    target_height = min(image1.shape[0], image2.shape[0]) # make both height and width the minimum of the two

    target_width = min(image1.shape[1], image2.shape[1])

    def cropper(image):

        """

        crops image to target height and width found in parent function and 
        returns cropped images
        
        
        """

        cut_off_top = (image.shape[0] - target_height) // 2

        cut_of_sides = (image.shape[1] - target_width) // 2

        if image.ndim == 2:
            return image[cut_off_top: target_height + cut_off_top, cut_of_sides: target_width + cut_of_sides]
        
        else:
            return image[cut_off_top: target_height + cut_off_top, cut_of_sides: target_width + cut_of_sides, : ]
    
    return cropper(image1), cropper(image2)


def hybrid_maker(image_low, image_high, low_size, low_sigma, high_size, high_sigma, x_shifted, y_shifted):

    """
    
    image_low will have its low frequencies added to imag_high's high frequencies. Gaussian

    filter applied to low and high images have been named in the function

    An alignment for the images will be needed, which will be done by eye measurement
    
    """

    image_low_to_use, image_high_to_use = image_matcher(image_low, image_high) # match the two images together so that they have same dimension

    if image_low_to_use.ndim == 3:   # color -> grayscale
        image_low_to_use = image_low_to_use.mean(axis=2).astype(np.float32)
    if image_high_to_use.ndim == 3:
        image_high_to_use = image_high_to_use.mean(axis=2).astype(np.float32)

    aligned_higher_image = translator(image_high_to_use, x_shifted, y_shifted) # align high filter image to match up with low filter image

    image_low_filtered = low_pass_filter(image_low_to_use, low_size, low_sigma) 

    image_high_filtered = high_pass_filter(aligned_higher_image, high_size, high_sigma)

    hybrid_image = np.clip(image_low_filtered + image_high_filtered, 0.0, 1.0).astype(np.float32)

    return image_low_filtered.astype(np.float32), image_high_filtered.astype(np.float32), hybrid_image, high_pass_filter(image_high_to_use, high_size, high_sigma) # i am returning multiple here for the FFT visualizations needed

def hybrid_maker_aligned(image_low, image_high_aligned,
                         low_size, low_sigma,
                         high_size, high_sigma):
    """
    Build a hybrid image from two images that are ALREADY aligned and same size.
    (No integer translate here; use align_images() beforehand.)
    Returns: (low_passed, high_passed, hybrid, high_passed_unaligned_debug)
    """
    import numpy as np

    # If they’re color, stay color; if grayscale, stay grayscale.
    # Your low/high filters already support both via per-channel helper.
    image_low  = image_low.astype(np.float32, copy=False)
    image_high_aligned = image_high_aligned.astype(np.float32, copy=False)

    # filter
    image_low_filtered  = low_pass_filter(image_low,  low_size,  low_sigma)
    image_high_filtered = high_pass_filter(image_high_aligned, high_size, high_sigma)

    hybrid_image = np.clip(image_low_filtered + image_high_filtered, 0.0, 1.0).astype(np.float32)

    # For FFT debug, return the high-pass of the *unaligned* high image = here equal to aligned
    return (image_low_filtered.astype(np.float32),
            image_high_filtered.astype(np.float32),
            hybrid_image,
            image_high_filtered.copy().astype(np.float32))


def fft_log_gray(image):

    """

    runs fft on image, moves low frequencies to center of image, then log scales 
    the image to make sure both brightest and darkest points can be seen clearly
    on same image
    
    
    """

    if (image.ndim == 3) and (image.shape[2] == 3): # in color images average the three channels

        gray_image = image.mean(axis = 2).astype(np.float32)

    else:

        gray_image = image.astype(np.float32)

    Fourier = np.fft.fft2(gray_image) # returns complex values representing magnitude and shifts of each sine and cosine curve that make up image

    shifted_Fourier = np.fft.fftshift(Fourier)

    log_magnitude = np.log(np.abs(shifted_Fourier) + 1e-8) # I only want magnitudes not shifts so sign does not matter. I use log scale to make sure too bright values do not distort rest of image

    # i added a tiny epsilon in case shifted fourier led to all values becoming zero this prevents log of zero error


    min_value, max_value = log_magnitude.min(), log_magnitude.max()

    if max_value - min_value < 1e-12: # if images are same color then avoid division by zero

        return np.zeros_like(gray_image, dtype = np.float32)
    
    return ((log_magnitude - min_value) / (max_value - min_value)).astype(np.float32)





























        



