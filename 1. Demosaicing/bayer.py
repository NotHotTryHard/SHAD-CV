import numpy as np
def get_bayer_masks(n_rows, n_cols):
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
        green_mask = np.tile(np.hstack((col_a, col_b)), n_cols // 2)
        blue_mask = np.tile(np.hstack((col_b, col_empty)), n_cols // 2)
        if n_cols % 2 == 1:
            red_mask = np.hstack((red_mask, col_empty))
            green_mask = np.hstack((green_mask, col_a))
            blue_mask = np.hstack((blue_mask, col_b))
        return np.dstack((red_mask, green_mask, blue_mask))


def get_colored_img(raw_img):
    """
    :param raw_img:
        `np.array` of shape `(n_rows, n_cols)` and dtype `np.uint8`,
        raw image

    :return:
        `np.array` of shape `(n_rows, n_cols, 3)` and dtype `np.uint8`,
        each channel contains known color values or zeros
        depending on Bayer masks
    """
    n_rows, n_cols = raw_img.shape
    masks = get_bayer_masks(n_rows, n_cols)
    red_mask, green_mask, blue_mask = masks[..., 0], masks[..., 1], masks[..., 2]
    return np.dstack((raw_img * red_mask, raw_img * green_mask, raw_img * blue_mask))


def get_raw_img(colored_img):
    """
    :param colored_img:
        `np.array` of shape `(n_rows, n_cols, 3)` and dtype `np.uint8`,
        colored image

    :return:
        `np.array` of shape `(n_rows, n_cols)` and dtype `np.uint8`,
        raw image as captured by camera
    """
    red_img, green_img, blue_img = colored_img[..., 0], colored_img[..., 1], colored_img[..., 2]
    n_rows, n_cols = red_img.shape
    masks = get_bayer_masks(n_rows, n_cols)
    red_mask, green_mask, blue_mask = masks[..., 0], masks[..., 1], masks[..., 2]
    return red_img * red_mask + green_img * green_mask + blue_img * blue_mask


def get_neighbor_offsets(n):
        offsets = []
        for dr in range(-n, n + 1):
            for dc in range(-n, n + 1):
                if dr != 0 or dc != 0:
                    offsets.append((dr, dc))
        return offsets


def bilinear_interpolation_slow(raw_img):
    """
    :param raw_img:
        `np.array` of shape `(n_rows, n_cols)` and dtype `np.uint8`,
        raw image

    :return:
        `np.array` of shape `(n_rows, n_cols, 3)`, and dtype `np.uint8`,
        result of bilinear interpolation
    """
    n_rows, n_cols = raw_img.shape
    
    masks = get_bayer_masks(n_rows, n_cols)
    red_mask, green_mask, blue_mask = masks[..., 0], masks[..., 1], masks[..., 2]
    
    images = get_colored_img(raw_img)
    red_image, green_image, blue_image = images[..., 0], images[..., 1], images[..., 2]

    data = []
    #red_data1 = []
    red_data2 = []
    for dc, dr in get_neighbor_offsets(1):
        data.append(np.roll(raw_img, (dr, dc), axis=(0, 1)))
        #red_data1.append(np.roll(red_image, (dr, dc), axis=(0, 1)))
        red_data2.append(np.roll(red_mask, (dr, dc), axis=(0, 1)))
    #red_layers = np.dstack(tuple(red_data1))
    layers = np.dstack(tuple(data))
    red_masks = np.dstack(tuple(red_data2))
    #red_interp = np.where(red_mask, raw_img, np.uint8(np.ma.mean(np.ma.masked_array(red_layers, mask=~red_masks), axis=-1)))
    red_interp = np.where(red_mask, raw_img, np.uint8(np.ma.mean(np.ma.masked_array(layers, mask=~red_masks), axis=-1)))

    #green_data1 = []
    green_data2 = []
    for dc, dr in get_neighbor_offsets(1):
        #green_data1.append(np.roll(green_image, (dr, dc), axis=(0, 1)))
        green_data2.append(np.roll(green_mask, (dr, dc), axis=(0, 1)))
    #green_layers = np.dstack(tuple(green_data1))
    green_masks = np.dstack(tuple(green_data2))
    #green_interp = np.where(green_mask, raw_img, np.uint8(np.ma.mean(np.ma.masked_array(green_layers, mask=~green_masks), axis=-1)))
    green_interp = np.where(green_mask, raw_img, np.uint8(np.ma.mean(np.ma.masked_array(layers, mask=~green_masks), axis=-1)))
    
    #blue_data1 = []
    blue_data2 = []
    for dc, dr in get_neighbor_offsets(1):
        #blue_data1.append(np.roll(blue_image, (dr, dc), axis=(0, 1)))
        blue_data2.append(np.roll(blue_mask, (dr, dc), axis=(0, 1)))
    #blue_layers = np.dstack(tuple(blue_data1))
    blue_masks = np.dstack(tuple(blue_data2))
    #blue_interp = np.where(blue_mask, raw_img, np.uint8(np.ma.mean(np.ma.masked_array(blue_layers, mask=~blue_masks), axis=-1)))
    blue_interp = np.where(blue_mask, raw_img, np.uint8(np.ma.mean(np.ma.masked_array(layers, mask=~blue_masks), axis=-1)))
    
    return np.dstack((red_interp, green_interp, blue_interp))


def bilinear_interpolation(raw_img):
    """
    :param raw_img:
        `np.array` of shape `(n_rows, n_cols)` and dtype `np.uint8`,
        raw image

    :return:
        `np.array` of shape `(n_rows, n_cols, 3)`, and dtype `np.uint8`,
        result of bilinear interpolation
    """
    n_rows, n_cols = raw_img.shape
    
    masks = get_bayer_masks(n_rows, n_cols)
    red_mask, green_mask, blue_mask = masks[..., 0], masks[..., 1], masks[..., 2]
    
    images = get_colored_img(raw_img)
    red_img, green_img, blue_img = images[..., 0], images[..., 1], images[..., 2]

    red_sum = np.zeros_like(raw_img, dtype=np.float64)
    green_sum = np.zeros_like(raw_img, dtype=np.float64)
    blue_sum = np.zeros_like(raw_img, dtype=np.float64)
    
    for dc, dr in get_neighbor_offsets(1):
        red_sum += np.roll(red_img, (dr, dc), axis=(0, 1))
        green_sum += np.roll(green_img, (dr, dc), axis=(0, 1))
        blue_sum += np.roll(blue_img, (dr, dc), axis=(0, 1))
    
    red_sum[:, ::2] /= 2
    red_sum[1::2, :] /= 2
    
    green_sum /= 4
    
    blue_sum[::2, :] /= 2
    blue_sum[:, 1::2] /= 2
    
    red_interp = np.where(red_mask, raw_img, np.uint8(red_sum))
    green_interp = np.where(green_mask, raw_img, np.uint8(green_sum))
    blue_interp = np.where(blue_mask, raw_img, np.uint8(blue_sum))

    return np.dstack((red_interp, green_interp, blue_interp))


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
        return np.dstack((red_mask, green_mask_red_row_blue_col, green_mask_red_col_blue_row, blue_mask))


def improved_interpolation_slow(raw_img):
    """
    :param raw_img:
        `np.array` of shape `(n_rows, n_cols)` and dtype `np.uint8`, raw image

    :return:
        `np.array` of shape `(n_rows, n_cols, 3)` and dtype `np.uint8`,
        result of improved interpolation
    """
    n_rows, n_cols = raw_img.shape
    
    masks = get_bayer_masks_improved(n_rows, n_cols)
    red_mask, green_mask_red_row_blue_col, green_mask_red_col_blue_row, blue_mask = masks[..., 0], masks[..., 1], masks[..., 2], masks[..., 3]
    green_mask = np.logical_or(green_mask_red_row_blue_col, green_mask_red_col_blue_row)
    red_blue_mask = np.logical_or(red_mask, blue_mask)   
    
    images = get_colored_img(raw_img)
    red_img, green_img, blue_img = images[..., 0], images[..., 1], images[..., 2]

    red_sum = np.zeros_like(raw_img, dtype=np.float32)
    green_sum = np.zeros_like(raw_img, dtype=np.float32)
    blue_sum = np.zeros_like(raw_img, dtype=np.float32)
    
    offsets = get_neighbor_offsets(1) + [(0, 2), (0, -2), (2, 0), (-2, 0), (0, 0)]
    
    for dr, dc in offsets:
        red_roll = np.float32(np.roll(red_img, (dr, dc), axis=(0, 1)))
        green_roll = np.float32(np.roll(green_img, (dr, dc), axis=(0, 1)))
        blue_roll = np.float32(np.roll(blue_img, (dr, dc), axis=(0, 1)))
        
        if (dr, dc) == (0, 0):
            red_sum[green_mask] += 5 * green_roll[green_mask]
            red_sum[blue_mask] += 6 * blue_roll[blue_mask]
            green_sum[red_mask] += 4 * red_roll[red_mask]
            green_sum[blue_mask] += 4 * blue_roll[blue_mask]
            blue_sum[green_mask] += 5 * green_roll[green_mask]
            blue_sum[red_mask] += 6 * red_roll[red_mask]
            
        elif (dr, dc) in {(0, 1), (0, -1)}:
            red_sum[green_mask_red_row_blue_col] += 4 * red_roll[green_mask_red_row_blue_col]
            green_sum[red_blue_mask] += 2 * green_roll[red_blue_mask]
            blue_sum[green_mask_red_col_blue_row] += 4 * blue_roll[green_mask_red_col_blue_row]
            
        elif (dr, dc) in {(1, 0), (-1, 0)}:
            red_sum[green_mask_red_col_blue_row] += 4 * red_roll[green_mask_red_col_blue_row]
            green_sum[red_blue_mask] += 2 * green_roll[red_blue_mask]
            blue_sum[green_mask_red_row_blue_col] += 4 * blue_roll[green_mask_red_row_blue_col]
        
        elif (dr, dc) in {(1, 1), (-1, 1), (-1, -1), (1, -1)}:
            red_sum[green_mask] -= 1 * green_roll[green_mask]
            red_sum[blue_mask] += 2 * red_roll[blue_mask]
            blue_sum[green_mask] -= 1 * green_roll[green_mask]
            blue_sum[red_mask] += 2 * blue_roll[red_mask]
        
        elif (dr, dc) in {(0, 2), (0, -2)}:
            green_sum[red_mask] -= 1 * red_roll[red_mask]
            green_sum[blue_mask] -= 1 * blue_roll[blue_mask]
            red_sum[green_mask_red_row_blue_col] -= 1 * green_roll[green_mask_red_row_blue_col]
            red_sum[green_mask_red_col_blue_row] += 1/2 * green_roll[green_mask_red_col_blue_row]
            red_sum[blue_mask] -= 3/2 * blue_roll[blue_mask]
            blue_sum[green_mask_red_col_blue_row] -= 1 * green_roll[green_mask_red_col_blue_row]
            blue_sum[green_mask_red_row_blue_col] += 1/2 * green_roll[green_mask_red_row_blue_col]
            blue_sum[red_mask] -= 3/2 * red_roll[red_mask]
            
        elif (dr, dc) in {(2, 0), (-2, 0)}:
            green_sum[red_mask] -= 1 * red_roll[red_mask]
            green_sum[blue_mask] -= 1 * blue_roll[blue_mask]
            red_sum[green_mask_red_row_blue_col] += 1/2 * green_roll[green_mask_red_row_blue_col]
            red_sum[green_mask_red_col_blue_row] -= 1 * green_roll[green_mask_red_col_blue_row]
            red_sum[blue_mask] -= 3/2 * blue_roll[blue_mask]
            blue_sum[green_mask_red_col_blue_row] += 1/2 * green_roll[green_mask_red_col_blue_row]
            blue_sum[green_mask_red_row_blue_col] -= 1 * green_roll[green_mask_red_row_blue_col]
            blue_sum[red_mask] -= 3/2 * red_roll[red_mask]

    red_interp = np.where(red_mask, raw_img, np.clip(red_sum / 8, 0, 255).astype(np.uint8))
    green_interp = np.where(green_mask, raw_img, np.clip(green_sum / 8, 0, 255).astype(np.uint8))
    blue_interp = np.where(blue_mask, raw_img, np.clip(blue_sum / 8, 0, 255).astype(np.uint8))

    
    return np.dstack((red_interp, green_interp, blue_interp))
        

def improved_interpolation(raw_img):
    """
    :param raw_img:
        `np.array` of shape `(n_rows, n_cols)` and dtype `np.uint8`, raw image

    :return:
        `np.array` of shape `(n_rows, n_cols, 3)` and dtype `np.uint8`,
        result of improved interpolation
    """
    n_rows, n_cols = raw_img.shape
    
    masks = get_bayer_masks_improved(n_rows, n_cols)
    red_mask, green_mask_red_row_blue_col, green_mask_red_col_blue_row, blue_mask = masks[..., 0], masks[..., 1], masks[..., 2], masks[..., 3]
    green_mask = np.logical_or(green_mask_red_row_blue_col, green_mask_red_col_blue_row)
    red_blue_mask = np.logical_or(red_mask, blue_mask)   
    
    images = get_colored_img(raw_img)
    red_img, green_img, blue_img = images[..., 0], images[..., 1], images[..., 2]

    red_sum = np.zeros_like(raw_img, dtype=np.float32)
    green_sum = np.zeros_like(raw_img, dtype=np.float32)
    blue_sum = np.zeros_like(raw_img, dtype=np.float32)
    
    red_sum_red_mask = red_sum[red_mask]
    red_sum_green_mask_red_row = red_sum[green_mask_red_row_blue_col]
    red_sum_green_mask_blue_row = red_sum[green_mask_red_col_blue_row]
    red_sum_blue_mask = red_sum[blue_mask]
    
    green_sum_red_mask = green_sum[red_mask]
    green_sum_green_mask_red_row = green_sum[green_mask_red_row_blue_col]
    green_sum_green_mask_blue_row = green_sum[green_mask_red_col_blue_row]
    green_sum_blue_mask = green_sum[blue_mask]
    
    blue_sum_red_mask = blue_sum[red_mask]
    blue_sum_green_mask_red_row = blue_sum[green_mask_red_row_blue_col]
    blue_sum_green_mask_blue_row = blue_sum[green_mask_red_col_blue_row]
    blue_sum_blue_mask = blue_sum[blue_mask]
    
    offsets = get_neighbor_offsets(1) + [(0, 2), (0, -2), (2, 0), (-2, 0), (0, 0)]
    
    for dr, dc in offsets:
        red_roll = np.float32(np.roll(red_img, (dr, dc), axis=(0, 1)))
        green_roll = np.float32(np.roll(green_img, (dr, dc), axis=(0, 1)))
        blue_roll = np.float32(np.roll(blue_img, (dr, dc), axis=(0, 1)))
        
        if (dr, dc) == (0, 0):
            red_sum_green_mask_blue_row += 5 * green_roll[green_mask_red_col_blue_row]
            red_sum_green_mask_red_row += 5 * green_roll[green_mask_red_row_blue_col]
            red_sum_blue_mask += 6 * blue_roll[blue_mask]
            green_sum_red_mask += 4 * red_roll[red_mask]
            green_sum_blue_mask += 4 * blue_roll[blue_mask]
            blue_sum_green_mask_red_row += 5 * green_roll[green_mask_red_row_blue_col]
            blue_sum_green_mask_blue_row += 5 * green_roll[green_mask_red_col_blue_row]
            blue_sum_red_mask += 6 * red_roll[red_mask]
            
        elif (dr, dc) in {(0, 1), (0, -1)}:
            red_sum_green_mask_red_row += 4 * red_roll[green_mask_red_row_blue_col]
            green_sum_red_mask += 2 * green_roll[red_mask]
            green_sum_blue_mask += 2 * green_roll[blue_mask]
            blue_sum_green_mask_blue_row += 4 * blue_roll[green_mask_red_col_blue_row]
            
        elif (dr, dc) in {(1, 0), (-1, 0)}:
            red_sum_green_mask_blue_row += 4 * red_roll[green_mask_red_col_blue_row]
            green_sum_red_mask += 2 * green_roll[red_mask]
            green_sum_blue_mask += 2 * green_roll[blue_mask]
            blue_sum_green_mask_red_row += 4 * blue_roll[green_mask_red_row_blue_col]
        
        elif (dr, dc) in {(1, 1), (-1, 1), (-1, -1), (1, -1)}:
            red_sum_green_mask_red_row -= 1 * green_roll[green_mask_red_row_blue_col]
            red_sum_green_mask_blue_row -= 1 * green_roll[green_mask_red_col_blue_row]
            red_sum_blue_mask += 2 * red_roll[blue_mask]
            blue_sum_green_mask_red_row -= 1 * green_roll[green_mask_red_row_blue_col]
            blue_sum_green_mask_blue_row -= 1 * green_roll[green_mask_red_col_blue_row]
            blue_sum_red_mask += 2 * blue_roll[red_mask]
        
        elif (dr, dc) in {(0, 2), (0, -2)}:
            green_sum_red_mask -= 1 * red_roll[red_mask]
            green_sum_blue_mask -= 1 * blue_roll[blue_mask]
            red_sum_green_mask_red_row -= 1 * green_roll[green_mask_red_row_blue_col]
            red_sum_green_mask_blue_row += 1/2 * green_roll[green_mask_red_col_blue_row]
            red_sum_blue_mask -= 3/2 * blue_roll[blue_mask]
            blue_sum_green_mask_blue_row -= 1 * green_roll[green_mask_red_col_blue_row]
            blue_sum_green_mask_red_row += 1/2 * green_roll[green_mask_red_row_blue_col]
            blue_sum_red_mask -= 3/2 * red_roll[red_mask]
            
        elif (dr, dc) in {(2, 0), (-2, 0)}:
            green_sum_red_mask -= 1 * red_roll[red_mask]
            green_sum_blue_mask -= 1 * blue_roll[blue_mask]
            red_sum_green_mask_red_row += 1/2 * green_roll[green_mask_red_row_blue_col]
            red_sum_green_mask_blue_row -= 1 * green_roll[green_mask_red_col_blue_row]
            red_sum_blue_mask -= 3/2 * blue_roll[blue_mask]
            blue_sum_green_mask_blue_row += 1/2 * green_roll[green_mask_red_col_blue_row]
            blue_sum_green_mask_red_row -= 1 * green_roll[green_mask_red_row_blue_col]
            blue_sum_red_mask -= 3/2 * red_roll[red_mask]
            
            
    red_sum[red_mask] = red_sum_red_mask
    red_sum[green_mask_red_row_blue_col] = red_sum_green_mask_red_row
    red_sum[green_mask_red_col_blue_row] = red_sum_green_mask_blue_row
    red_sum[blue_mask] = red_sum_blue_mask
    
    green_sum[red_mask] = green_sum_red_mask
    green_sum[green_mask_red_row_blue_col] = green_sum_green_mask_red_row
    green_sum[green_mask_red_col_blue_row] = green_sum_green_mask_blue_row
    green_sum[blue_mask] = green_sum_blue_mask
    
    blue_sum[red_mask] = blue_sum_red_mask
    blue_sum[green_mask_red_row_blue_col] = blue_sum_green_mask_red_row
    blue_sum[green_mask_red_col_blue_row] = blue_sum_green_mask_blue_row
    blue_sum[blue_mask] = blue_sum_blue_mask 

    red_interp = np.where(red_mask, raw_img, np.clip(red_sum / 8, 0, 255).astype(np.uint8))
    green_interp = np.where(green_mask, raw_img, np.clip(green_sum / 8, 0, 255).astype(np.uint8))
    blue_interp = np.where(blue_mask, raw_img, np.clip(blue_sum / 8, 0, 255).astype(np.uint8))

    
    return np.dstack((red_interp, green_interp, blue_interp))


def compute_mse(img_pred, img_gt):
    n_rows, n_cols, channels = img_pred.shape
    
    return np.sum((np.float64(img_pred) - np.float64(img_gt)) ** 2) / channels / n_cols / n_rows


def compute_psnr(img_pred, img_gt):
    """
    :param img_pred:
        `np.array` of shape `(n_rows, n_cols, 3)` and dtype `np.uint8`,
        predicted image
    :param img_gt:
        `np.array` of shape `(n_rows, n_cols, 3)` and dtype `np.uint8`,
        ground truth image

    :return:
        `float`, PSNR metric
    """
    mse = compute_mse(img_pred, img_gt)
    if not mse:
        raise ValueError
    return 10 * np.log10(np.max(np.float64(img_gt)) ** 2 / mse) 


if __name__ == "__main__":
    from PIL import Image

    raw_img_path = "tests/04_unittest_bilinear_img_input/02.png"
    raw_img = np.array(Image.open(raw_img_path))

    img_bilinear = bilinear_interpolation(raw_img)
    Image.fromarray(img_bilinear).save("bilinear.png")

    img_improved = improved_interpolation(raw_img)
    Image.fromarray(img_improved).save("improved.png")
