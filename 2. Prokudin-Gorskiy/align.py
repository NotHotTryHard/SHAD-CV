import numpy as np
from PIL import Image
# Read the implementation of the align_image function in pipeline.py
# to see, how these functions will be used for image alignment.


def extract_channel_plates(raw_img, crop):
    # Your code here
    dh, dw = 0, 0
    n_rows, n_cols = raw_img.shape
    height = n_rows // 3
    width = n_cols
    
    blue_img = raw_img[0: height, :]
    green_img = raw_img[height: 2 * height, :]
    red_img = raw_img[2 * height: 3 * height, :]
    
    if crop:
        dh = int(height / 10)
        dw = int(width / 10)
        blue_img = blue_img[dh: height - dh, dw: width - dw]
        green_img = green_img[dh: height - dh, dw: width - dw]
        red_img = red_img[dh: height - dh, dw: width - dw]
    
    unaligned_rgb = (red_img, green_img, blue_img)
    coords = (np.array([2 * height + dh, dw]), np.array([height + dh, dw]), np.array([dh, dw]))
    return unaligned_rgb, coords


def compute_mse(img_1, img_2):
    n_rows, n_cols = img_1.shape
    return np.sum((np.float64(img_1) - np.float64(img_2)) ** 2) / n_cols / n_rows


def compute_cross_correlation(img_1, img_2):
    return np.sum(img_1 * img_2) / np.sqrt(np.sum(np.power(img_1, 2)) * np.sum(np.power(img_2, 2)))


def compute_mse_matrix(shifted_a, shifted_b):
    k, k, n_rows, n_cols = shifted_a.shape
    return np.sum((np.float64(shifted_a) - np.float64(shifted_b)) ** 2, axis=(-2, -1)) / n_cols / n_rows


def compute_cross_correlation_martix(shifted_a, shifted_b):
    shifted_a = np.float64(shifted_a)
    shifted_b = np.float64(shifted_b)
    return np.sum(shifted_a * shifted_b, axis=(-2, -1)) / np.sqrt(np.sum(np.power(shifted_a, 2), axis=(-2, -1)) * np.sum(np.power(shifted_b, 2), axis=(-2, -1)))


def get_bayer_masks_improved(n_rows, n_cols):
    """
    :param n_rows: `int`, number of rows
    :param n_cols: `int`, number of columns

    :return:
        `np.array` of shape `(n_rows, n_cols, 3)` and dtype `np.bool_`
        containing red, green and blue Bayer masks
    """
    col_empty = np.zeros(n_rows, dtype=np.bool_)[:, np.newaxis]
    col_a = np.zeros(n_rows, dtype=np.bool_)[:, np.newaxis]
    col_a[0::2] = True
    col_b = np.zeros(n_rows, dtype=np.bool_)[:, np.newaxis]
    col_b[1::2] = True

    if n_cols == 1:
        return np.dstack((col_empty, col_a, col_b))
    else:
        red_mask = np.tile(np.hstack((col_empty, col_a)), n_cols // 2)
        green_mask_red_row_blue_col = np.tile(np.hstack((col_a, col_empty)), n_cols // 2)
        green_mask_red_col_blue_row = np.tile(np.hstack((col_empty, col_b)), n_cols // 2)
        blue_mask = np.tile(np.hstack((col_b, col_empty)), n_cols // 2)
        if n_cols % 2 == 1:
            red_mask = np.hstack((red_mask, col_empty))
            
            green_mask_red_row_blue_col = np.hstack((green_mask_red_row_blue_col, col_a))
            green_mask_red_col_blue_row = np.hstack((green_mask_red_col_blue_row, col_empty))
            
            blue_mask = np.hstack((blue_mask, col_b))
        return (red_mask, green_mask_red_row_blue_col, green_mask_red_col_blue_row, blue_mask)

def downsample_slow(img):
    height, width = img.shape
    height -= height % 2
    width -= width % 2
    mask_a, mask_b, mask_c, mask_d = get_bayer_masks_improved(height, width)
    return np.uint8((img[:height, :width][mask_a] + img[:height, :width][mask_b] + img[:height, :width][mask_c] + img[:height, :width][mask_d]).reshape(height // 2, width // 2) / 4)

#позаимствовано с https://stackoverflow.com/questions/34231244/downsampling-a-2d-numpy-array-in-pythons
def downsample(img):
    """
        Downsamples an ndarray of size `(h,w)` along axes 0,1
        Input can be non-float, e.g. uint8
    """
    dtype1 = img.dtype
    new_img = img.astype(float)
    
    height, width = new_img.shape
    
    if height % 2 == 1 and width % 2 == 0:
        new_img = new_img[:-1, :]
    elif height % 2 == 1 and width % 2 == 1:
        new_img = new_img[:-1, :-1]
    elif height % 2 == 0 and width % 2 == 1:
        new_img = new_img[:, :-1]
        
    height, width = new_img.shape
    new_width = width // 2
    new_height = height // 2
    new_img = new_img.reshape((height, new_width, 2))
    new_img = np.mean(new_img, axis=2)
    assert new_img.shape == (height, new_width)
    new_img = new_img.reshape((new_height, 2, new_width))
    new_img = np.mean(new_img, axis=1)
    assert new_img.shape == (new_height, new_width)
    new_img = new_img.astype(dtype1)
    return new_img


def create_shifted_img(img, boundary):
    height, width = img.shape
    shifted_img = np.zeros((2 * boundary + 1, 2 * boundary + 1, height, width), dtype=img.dtype)

    for i in range(2 * boundary + 1):
        for j in range(2 * boundary + 1):
            shifted_img[i, j] = np.roll(np.roll(img, shift=i-boundary, axis=0), shift=j-boundary, axis=1)

    return shifted_img


def create_matrix_images(img, boundary):
    # Получаем размеры исходной матрицы
    n = 2 * boundary + 1
    img_shape = img.shape

    result_matrix = np.tile(img.flatten(), (n, n))
    result_matrix = result_matrix.reshape(n, n, *img_shape)

    return result_matrix


def find_relative_shift_pyramid(img_a, img_b):
    # Your code here
    lower_level_bound = 15
    upper_level_bound = 2
    height, width = img_a.shape
    
    pyramid_a = [img_a]
    pyramid_b = [img_b]
    
    while height > 500 or width > 500:
        pyramid_a.append(downsample(pyramid_a[-1]))
        pyramid_b.append(downsample(pyramid_b[-1]))
        height, width = pyramid_a[-1].shape
    
    curr_level_a = pyramid_a.pop()
    curr_level_b = pyramid_b.pop()
    
    shifted_a = create_shifted_img(curr_level_a, lower_level_bound)
    unshifted_b = create_matrix_images(curr_level_b, lower_level_bound)
    
    metrics = compute_mse_matrix(shifted_a, unshifted_b)
    if np.max(metrics) == 0:
        shift = np.array([0, 0])
    else:
        shift = np.unravel_index(np.argmin(metrics, axis=None), metrics.shape)
        shift -= np.array([lower_level_bound, lower_level_bound])
    
    while pyramid_a:
        curr_level_a = pyramid_a.pop()
        curr_level_b = pyramid_b.pop()
        
        shifted_a = create_shifted_img(curr_level_a, upper_level_bound)
        unshifted_b = create_matrix_images(curr_level_b, upper_level_bound)
        
        metrics = compute_mse_matrix(shifted_a, unshifted_b)
        if np.max(metrics) == 0:
            cur_shift = np.array([0, 0])
        else:
            cur_shift = np.unravel_index(np.argmin(metrics, axis=None), metrics.shape)
            cur_shift -= np.array([upper_level_bound, upper_level_bound])
        shift = shift * 2 + cur_shift
        
    a_to_b = np.array(shift)
    return a_to_b

def find_relative_shift_layers_slow(img_a, img_b, boundary):
    # Your code here
    height, width = img_a.shape
    
    shifted_a = create_shifted_img(img_a, boundary)
    unshifted_b = create_matrix_images(img_b, boundary)
    
    metrics = compute_mse_matrix(shifted_a, unshifted_b)
    if np.max(metrics) == 0:
        shift = np.array([0, 0])
    else:
        shift = np.unravel_index(np.argmin(metrics, axis=None), metrics.shape)
        shift -= np.array([boundary, boundary])
        
    a_to_b = np.array(shift)
    return a_to_b

def find_relative_shift_layers_fast(img_a, img_b, boundary=15):
    # Your code here
    height, width = img_a.shape

    mse_max = -1
    mse_min = np.inf
    shift = [0, 0]
    for i in range(2 * boundary + 1):
        for j in range(2 * boundary + 1):
            shifted_img = np.roll(np.roll(img_a, shift=i-boundary, axis=0), shift=j-boundary, axis=1)
            mse = np.sum((np.float64(shifted_img) - np.float64(img_b)) ** 2) / (height * width)
            mse_max = max(mse, mse_max)
            if mse < mse_min:
                mse_min = mse
                shift = [i - boundary, j - boundary]
    if mse_max == 0:
        shift = [0, 0]
        
    a_to_b = np.array(shift)
    return a_to_b


def find_relative_shift_pyramid(img_a, img_b):
    # Your code here
    calibration_boundary = 15
    correction_boundary = 1
    height, width = img_a.shape
    
    pyramid_a = [img_a]
    pyramid_b = [img_b]

    while height > 200 or width > 200:
        pyramid_a.append(downsample(pyramid_a[-1]))
        pyramid_b.append(downsample(pyramid_b[-1]))
        height, width = pyramid_a[-1].shape
        
    #calibration
    curr_level_a = pyramid_a.pop()
    curr_level_b = pyramid_b.pop()
    
    shift = find_relative_shift_layers_fast(curr_level_a, curr_level_b, calibration_boundary)
    while pyramid_a:
        curr_level_a = pyramid_a.pop()
        curr_level_b = pyramid_b.pop()
        shift *= 2
        shift += find_relative_shift_layers_fast(np.roll(curr_level_a, shift.astype(tuple), axis=(0, 1)), curr_level_b, correction_boundary)

        
    a_to_b = np.array(shift)
    return a_to_b


def find_absolute_shifts(
    crops,
    crop_coords,
    find_relative_shift_fn,
):
    # Your code here
    
    r_to_g = find_relative_shift_fn(crops[0], crops[1]) + crop_coords[1] - crop_coords[0]
    b_to_g = find_relative_shift_fn(crops[2], crops[1]) + crop_coords[1] - crop_coords[2]
    return r_to_g, b_to_g


def create_aligned_image(
    channels,
    channel_coords,
    r_to_g,
    b_to_g,
):
    # Your code here
    height, width = channels[1].shape
    
    rel_r_to_g = r_to_g - channel_coords[1] + channel_coords[0]
    rel_b_to_g = b_to_g - channel_coords[1] + channel_coords[2]
    
    trim_up = max([rel_r_to_g[0], rel_b_to_g[0], 0])
    trim_down = -min([rel_r_to_g[0], rel_b_to_g[0], 0])
    trim_left = max([rel_r_to_g[1], rel_b_to_g[1], 0])
    trim_right = -min([rel_r_to_g[1], rel_b_to_g[1], 0])
    
    
    green_img = channels[1][trim_up: height - trim_down, trim_left: width - trim_right]
    red_img = np.roll(channels[0], tuple(rel_r_to_g), axis=(0, 1))[trim_up: height - trim_down, trim_left: width - trim_right]
    blue_img = np.roll(channels[2], tuple(rel_b_to_g), axis=(0, 1))[trim_up: height - trim_down, trim_left: width - trim_right]
    aligned_img = np.dstack((red_img, green_img, blue_img))
    return aligned_img


def find_relative_shift_fourier(img_a, img_b):
    # Your code here
    cross_correlation = np.fft.ifft2(np.conj(np.fft.fft2(img_a)) * np.fft.fft2(img_b))

    shift = np.unravel_index(np.argmax(np.abs(cross_correlation)), img_a.shape)
    shifts = np.array(shift)
    
    if shifts[0] > img_a.shape[0] // 2:
        shifts[0] -= img_a.shape[0]
    if shifts[1] > img_a.shape[1] // 2:
        shifts[1] -= img_a.shape[1]

    return shifts


if __name__ == "__main__":
    import common
    import pipeline

    # Read the source image and the corresponding ground truth information
    test_path = "tests/05_unittest_align_image_pyramid_img_small_input/00"
    raw_img, (r_point, g_point, b_point) = common.read_test_data(test_path)

    # Draw the same point on each channel in the original
    # raw image using the ground truth coordinates
    visualized_img = pipeline.visualize_point(raw_img, r_point, g_point, b_point)
    common.save_image(f"gt_visualized.png", visualized_img)

    for method in ["pyramid", "fourier"]:
        # Run the whole alignment pipeline
        r_to_g, b_to_g, aligned_img = pipeline.align_image(raw_img, method)
        common.save_image(f"{method}_aligned.png", aligned_img)

        # Draw the same point on each channel in the original
        # raw image using the predicted r->g and b->g shifts
        # (Compare with gt_visualized for debugging purposes)§
        r_pred = g_point - r_to_g
        b_pred = g_point - b_to_g
        visualized_img = pipeline.visualize_point(raw_img, r_pred, g_point, b_pred)

        r_error = abs(r_pred - r_point)
        b_error = abs(b_pred - b_point)
        print(f"{method}: {r_error = }, {b_error = }")

        common.save_image(f"{method}_visualized.png", visualized_img)
