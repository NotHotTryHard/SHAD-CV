import io
import pickle
import zipfile

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.io import imread
from skimage.metrics import peak_signal_noise_ratio

# !Этих импортов достаточно для решения данного задания, нельзя использовать другие библиотеки!


def pca_compression(matr, p):
    """Сжатие изображения с помощью PCA
    Вход: двумерная матрица (одна цветовая компонента картинки), количество компонент
    Выход: собственные векторы, проекция матрицы на новое пр-во и средние значения до центрирования
    """

    # Your code here
    matr_mean = np.mean(matr.T, axis = 0)
    matr_normalised = matr.T - matr_mean
    
    # Найдем матрицу ковариации
    cov = np.cov(matr_normalised, rowvar = False)
    # Ищем собственные значения и собственные векторы матрицы ковариации, используйте linalg.eigh из numpy
    eigenVals, eigenVecs = np.linalg.eigh(cov)
    # Посчитаем количество найденных собственных векторов
    #count = np.size(eigenVals)
    # Сортируем собственные значения в порядке убывания
    idx = np.argsort(eigenVals)[::-1]
    # Сортируем собственные векторы согласно отсортированным собственным значениям
    # !Это все для того, чтобы мы производили проекцию в направлении максимальной дисперсии!
    # Оставляем только p собственных векторов
    eigenVecs = eigenVecs[:, idx][:, :p]
    #eigenVals = eigenVals[idx][:p]
    # Проекция данных на новое пространство
    matr_reduced = np.dot(eigenVecs.T, matr_normalised.T)
    return eigenVecs, matr_reduced, matr_mean


def pca_decompression(compressed):
    """Разжатие изображения
    Вход: список кортежей из собственных векторов и проекций для каждой цветовой компоненты
    Выход: разжатое изображение
    """

    result_img = []
    for i, comp in enumerate(compressed):
        # Матрично умножаем собственные векторы на проекции и прибавляем среднее значение по строкам исходной матрицы
        # !Это следует из описанного в самом начале примера!

        # Your code here
        result_img.append(np.clip(np.dot(comp[0], comp[1]) + comp[2][:, None], 0, 255))
        
    return np.dstack((result_img[0], result_img[1], result_img[2]))


def pca_visualize():
    plt.clf()
    img = imread("cat.png")
    if len(img.shape) == 3:
        img = img[..., :3]
    fig, axes = plt.subplots(3, 3)
    fig.set_figwidth(12)
    fig.set_figheight(12)

    for i, p in enumerate([1, 5, 10, 20, 50, 100, 150, 200, 256]):
        compressed = []
        for j in range(0, 3):
            # Your code here
            eigenVecs, matr_reduced, matr_mean = pca_compression(img[:, :, j], p)
            compressed.append((eigenVecs, matr_reduced, matr_mean))
        img_compressed = pca_decompression(compressed) / 255
        axes[i // 3, i % 3].imshow(img_compressed)
        axes[i // 3, i % 3].set_title("Компонент: {}".format(p))

    fig.savefig("pca_visualization.png")


def rgb2ycbcr(img_rgb):
    """Переход из пр-ва RGB в пр-во YCbCr
    Вход: RGB изображение
    Выход: YCbCr изображение
    """
    shape = img_rgb.shape
    
    coefs = np.array([[0.299, 0.587, 0.114],
                     [-0.1687, -0.3313, 0.5],
                     [0.5, -0.4187, -0.0813]])
    shift = np.array([0, 128, 128])[:, None]
    img_ycbcr = (np.dot(coefs, img_rgb.reshape(-1, 3).T) + shift).T.reshape(shape)

    return img_ycbcr


def ycbcr2rgb(img_ycbcr):
    """Переход из пр-ва YCbCr в пр-во RGB
    Вход: YCbCr изображение
    Выход: RGB изображение
    """
    shape = img_ycbcr.shape
    
    coefs = np.array([[1, 0, 1.402],
                     [1, -0.34414, -0.71414],
                     [1, 1.77, 0]])
    shift = np.array([0, 128, 128])[:, None]
    img_rgb = np.dot(coefs, (img_ycbcr.reshape(-1, 3).T - shift)).T.reshape(shape)

    return img_rgb


def get_gauss_1():
    sigma = 4
    plt.clf()
    rgb_img = imread("Lenna.png")
    if len(rgb_img.shape) == 3:
        rgb_img = rgb_img[..., :3]
    
    img_ycbcr = rgb2ycbcr(rgb_img)
    img_ycbcr[:, :, 1] = gaussian_filter(img_ycbcr[:, :, 1], sigma=sigma)
    img_ycbcr[:, :, 2] = gaussian_filter(img_ycbcr[:, :, 2], sigma=sigma)
    img_ycbcr = np.clip(img_ycbcr, 0, 255)
    
    rgb_img = np.clip(ycbcr2rgb(img_ycbcr), 0, 255).astype(np.uint8)
    plt.imshow(rgb_img)
    plt.savefig("gauss_1.png")


def get_gauss_2():
    sigma = 4
    plt.clf()
    rgb_img = imread("Lenna.png")
    if len(rgb_img.shape) == 3:
        rgb_img = rgb_img[..., :3]
        
    img_ycbcr = rgb2ycbcr(rgb_img)
    img_ycbcr[:, :, 0] = gaussian_filter(img_ycbcr[:, :, 0], sigma=sigma)
    img_ycbcr = np.clip(img_ycbcr, 0, 255)
    
    rgb_img = np.clip(ycbcr2rgb(img_ycbcr), 0, 255).astype(np.uint8)
    plt.imshow(rgb_img)
    plt.savefig("gauss_2.png")


def downsampling(component, sigma=10):
    """Уменьшаем цветовые компоненты в 2 раза
    Вход: цветовая компонента размера [A, B]
    Выход: цветовая компонента размера [A // 2, B // 2]
    """
    filtered_component = gaussian_filter(component, sigma=sigma)
    return filtered_component[::2, ::2]


def dct(block):
    """Дискретное косинусное преобразование
    Вход: блок размера 8x8
    Выход: блок размера 8x8 после ДКП
    """
    G = np.ones((8, 8))
    G[0,:] = 1 / np.sqrt(2)
    G[:,0] = 1 / np.sqrt(2)
    G[0][0] = 1 / 2
    G /= 4

    xv, uv = np.meshgrid(np.arange(8), np.arange(8))
    table = np.cos((2 * xv + 1) * uv * np.pi / 16)

    for u in range(8):
        for v in range(8):
            sum = 0
            for i in range(8):
                for j in range(8):
                    sum += block[i][j] * table[u][i] * table[v][j]
            G[u][v] *= sum
    return G


# Матрица квантования яркости
y_quantization_matrix = np.array(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ]
)

# Матрица квантования цвета
color_quantization_matrix = np.array(
    [
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
    ]
)


def quantization(block, quantization_matrix):
    """Квантование
    Вход: блок размера 8x8 после применения ДКП; матрица квантования
    Выход: блок размера 8x8 после квантования. Округление осуществляем с помощью np.round
    """

    return np.round(block / quantization_matrix)


def own_quantization_matrix(default_quantization_matrix, q):
    """Генерация матрицы квантования по Quality Factor
    Вход: "стандартная" матрица квантования; Quality Factor
    Выход: новая матрица квантования
    Hint: если после проделанных операций какие-то элементы обнулились, то замените их единицами
    """

    assert 1 <= q <= 100

    if q < 50:
        S = 5000 / q
    elif q <= 99:
        S = 200 - 2 * q
    else:
        S = 1
        
    new_quantization_matrix = np.floor((S * default_quantization_matrix + 50) / 100)
    new_quantization_matrix[new_quantization_matrix == 0] = 1
    return new_quantization_matrix


def zigzag(block):
    """Зигзаг-сканирование
    Вход: блок размера 8x8
    Выход: список из элементов входного блока, получаемый после его обхода зигзаг-сканированием
    """
    zigzag_rows = np.array([0, 0, 1, 2, 1, 0, 0, 1, 2, 3, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 4, 5, 6, 7, 7, 6, 5, 6, 7, 7])
    zigzag_cols = np.array([0, 1, 0, 0, 1, 2, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 5, 6, 7, 7, 6, 7])

    return block[zigzag_rows, zigzag_cols]


def compression(zigzag_list):
    """Сжатие последовательности после зигзаг-сканирования
    Вход: список после зигзаг-сканирования
    Выход: сжатый список в формате, который был приведен в качестве примера в самом начале данного пункта
    """

    new_list = []
    count = 0
    for k in zigzag_list:
        if k != 0:
            if count > 0:
                new_list.append(0)
                new_list.append(count)
                count = 0
            new_list.append(k)
        else:
            count += 1
    if count > 0:
        new_list.append(0)
        new_list.append(count)
            
    return np.array(new_list)


def pad_to_multiple_of_8(img_component):
    A, B = img_component.shape

    pad_A = (8 - A % 8) % 8
    pad_B = (8 - B % 8) % 8
    pad_width = ((0, pad_A), (0, pad_B))
    padded_img = np.pad(img_component, pad_width, mode='constant', constant_values=0)

    return padded_img


def jpeg_compression(img_rgb, quantization_matrixes, sigma=10):
    """JPEG-сжатие
    Вход: цветная картинка, список из 2-ух матриц квантования
    Выход: список списков со сжатыми векторами: [[compressed_y1,...], [compressed_Cb1,...], [compressed_Cr1,...]]
    """

    # Your code here

    # Переходим из RGB в YCbCr
    img_ycbcr = rgb2ycbcr(img_rgb)
    # Уменьшаем цветовые компоненты
    img_y = img_ycbcr[:, :, 0] - 128
    #print('Real: ', img_ycbcr[:, :, 1][:16, :16])
    img_cb_compr = downsampling(img_ycbcr[:, :, 1], sigma) - 128
    #print('Downsampled: ', img_cb_compr[:8, :8])
    img_cr_compr = downsampling(img_ycbcr[:, :, 2], sigma) - 128
    # Делим все компоненты на блоки 8x8 и все элементы блоков переводим из [0, 255] в [-128, 127]
    y_list = []
    cb_list = []
    cr_list = []
    
    img_y = pad_to_multiple_of_8(img_y)
    height, width = img_y.shape
    for i in range(height // 8):
        for j in range(width // 8):
            block = img_y[8 * i: 8 * (i + 1), 8 * j: 8 * (j + 1)]
            G = dct(block)  # Применяем ДКП, квантование, зизгаз-сканирование и сжатие
            G_quantized = quantization(G, quantization_matrixes[0])
            G_zigzag = zigzag(G_quantized)
            G_compressed = compression(G_zigzag)
            '''if i == 3 and j == 4:
                print('block: ', block)
                print('G: ', G)
                print('G_quantized: ', G_quantized)
                print('G_zigzag: ', G_zigzag)
                print('G_compressed: ', G_compressed)'''
            y_list.append(G_compressed)
            
    img_cb_compr = pad_to_multiple_of_8(img_cb_compr)
    height, width = img_cb_compr.shape
    for i in range(height // 8):
        for j in range(width // 8):
            block = img_cb_compr[8 * i: 8 * (i + 1), 8 * j: 8 * (j + 1)]
            G = dct(block)  # Применяем ДКП, квантование, зизгаз-сканирование и сжатие
            G_quantized = quantization(G, quantization_matrixes[1])
            G_zigzag = zigzag(G_quantized)
            G_compressed = compression(G_zigzag)
            cb_list.append(G_compressed)
            
    img_cr_compr = pad_to_multiple_of_8(img_cr_compr)
    height, width = img_cr_compr.shape
    for i in range(height // 8):
        for j in range(width // 8):
            block = img_cr_compr[8 * i: 8 * (i + 1), 8 * j: 8 * (j + 1)]
            G = dct(block)  # Применяем ДКП, квантование, зизгаз-сканирование и сжатие
            G_quantized = quantization(G, quantization_matrixes[1])
            G_zigzag = zigzag(G_quantized)
            G_compressed = compression(G_zigzag)
            cr_list.append(G_compressed)

    return [y_list, cb_list, cr_list]


def inverse_compression(compressed_list):
    """Разжатие последовательности
    Вход: сжатый список
    Выход: разжатый список
    """
    decompressed_list = []
    prev = -1
    for k in compressed_list:
        if prev == 0:
            for _ in range(np.uint(k)):
                decompressed_list.append(0)
            prev = -1
        elif k:
            decompressed_list.append(k)
        prev = k
    return decompressed_list


def inverse_zigzag(input):
    """Обратное зигзаг-сканирование
    Вход: список элементов
    Выход: блок размера 8x8 из элементов входного списка, расставленных в матрице в порядке их следования в зигзаг-сканировании
    """

    inverse_zigzag_ind = np.array([ 0,  1,  5,  6, 14, 15, 27, 28,  2,  4,  7, 13, 16, 26, 29, 42,  3,
        8, 12, 17, 25, 30, 41, 43,  9, 11, 18, 24, 31, 40, 44, 53, 10, 19,
       23, 32, 39, 45, 52, 54, 20, 22, 33, 38, 46, 51, 55, 60, 21, 34, 37,
       47, 50, 56, 59, 61, 35, 36, 48, 49, 57, 58, 62, 63])

    inverse_zigzag = input[inverse_zigzag_ind].reshape(8, 8)
    return inverse_zigzag


def inverse_quantization(block, quantization_matrix):
    """Обратное квантование
    Вход: блок размера 8x8 после применения обратного зигзаг-сканирования; матрица квантования
    Выход: блок размера 8x8 после квантования. Округление не производится
    """

    return block * quantization_matrix


def inverse_dct(block):
    """Обратное дискретное косинусное преобразование
    Вход: блок размера 8x8
    Выход: блок размера 8x8 после обратного ДКП. Округление осуществляем с помощью np.round
    """
    f = np.zeros((8, 8)) 
    
    alpha = np.ones(8)
    alpha[0] = 1 / np.sqrt(2)

    xv, uv = np.meshgrid(np.arange(8), np.arange(8))
    table = np.cos((2 * xv + 1) * uv * np.pi / 16)

    for i in range(8):
        for j in range(8):
            sum = 0
            for u in range(8):
                for v in range(8):
                    alpha[u] * alpha[v]
                    block[u, v]
                    table[u][i] * table[v][j]
                    sum += alpha[u] * alpha[v] * block[u, v] * table[u][i] * table[v][j]
            f[i][j] = sum / 4
    return np.round(f)


def upsampling(component):
    """Увеличиваем цветовые компоненты в 2 раза
    Вход: цветовая компонента размера [A, B, 1]
    Выход: цветовая компонента размера [2 * A, 2 * B, 1]
    """

    return np.repeat(np.repeat(component, 2, axis=0), 2, axis=1)


def jpeg_decompression(result, res_shape, quantization_matrixes):
    """Разжатие изображения
    Вход: result список сжатых данных, размер ответа, список из 2-ух матриц квантования
    Выход: разжатое изображение
    """
    result_shape = np.array(res_shape)
    y_list, cb_list, cr_list = result
    y_shape = np.ceil(result_shape / 8).astype(np.uint)
    img_y = np.zeros((y_shape[0] * 8, y_shape[1] * 8))
    
    for i in range(y_shape[0]):
        for j in range(y_shape[1]):
            g_compressed = y_list[i * y_shape[0] + j]
            g_zigzag = np.array(inverse_compression(g_compressed))
            g_quantized = inverse_zigzag(g_zigzag)
            g = inverse_quantization(g_quantized, quantization_matrixes[0])
            F = inverse_dct(g)
            img_y[8 * i: 8 * (i + 1), 8 * j: 8 * (j + 1)] = F + 128
            '''if i == 3 and j == 4:
                print('g_compressed: ', g_compressed)
                print('g_zigzag: ', g_zigzag)
                print('g_quantized: ', g_quantized)
                print('g: ', g)
                print('F: ', F)'''
    
    result_shape_min = result_shape // 2
    cb_shape = np.ceil(result_shape_min / 8).astype(np.uint)
    img_cb = np.zeros((result_shape_min[0], result_shape_min[1]))
    img_cr = np.zeros((result_shape_min[0], result_shape_min[1]))
    
    for i in range(cb_shape[0]):
        for j in range(cb_shape[1]):
            g_compressed = cb_list[i * cb_shape[0] + j]
            g_zigzag = np.array(inverse_compression(g_compressed))
            g_quantized = inverse_zigzag(g_zigzag)
            g = inverse_quantization(g_quantized, quantization_matrixes[1])
            F = inverse_dct(g)
            img_cb[8 * i: 8 * (i + 1), 8 * j: 8 * (j + 1)] = F + 128
            
            g_compressed = cr_list[i * cb_shape[0] + j]
            g_zigzag = np.array(inverse_compression(g_compressed))
            g_quantized = inverse_zigzag(g_zigzag)
            g = inverse_quantization(g_quantized, quantization_matrixes[1])
            F = inverse_dct(g)
            img_cr[8 * i: 8 * (i + 1), 8 * j: 8 * (j + 1)] = F + 128
    #print('Downsampled decompressed: ', img_cb[:8, :8])
    img_cb_norm = upsampling(img_cb)
    #print('Decompressed: ', img_cb_norm[:16, :16])
    img_cr_norm = upsampling(img_cr)
    
    img_ycbcr = np.zeros((result_shape[0], result_shape[1], 3))
    img_ycbcr[:, :, 0] = img_y[0: result_shape[0], 0: result_shape[1]]
    img_ycbcr[:, :, 1] = img_cb_norm[0: result_shape[0], 0: result_shape[1]]
    img_ycbcr[:, :, 2] = img_cr_norm[0: result_shape[0], 0: result_shape[1]]
    
    #print('Y: ', np.min(img_ycbcr[:, :, 0]), np.max(img_ycbcr[:, :, 0]))
    #print('Cb: ', np.min(img_ycbcr[:, :, 1]), np.max(img_ycbcr[:, :, 1]))
    #print('Cr: ', np.min(img_ycbcr[:, :, 2]), np.max(img_ycbcr[:, :, 2]))
    
    
    img_rgb = ycbcr2rgb(img_ycbcr)
    
    return np.clip(img_rgb, 0, 255).astype(np.uint8)


def jpeg_visualize():
    plt.clf()
    img = imread("Lenna.png")
    if len(img.shape) == 3:
        img = img[..., :3]
    fig, axes = plt.subplots(2, 3)
    fig.set_figwidth(12)
    fig.set_figheight(12)

    for i, p in enumerate([1, 10, 20, 50, 80, 100]):
        # Your code here
        print(f'Now running for the {p} value of parameter.')
        y_quantization_matrix_mine = own_quantization_matrix(y_quantization_matrix, p)
        color_quantization_matrix_mine = own_quantization_matrix(color_quantization_matrix, p)
        quantization_matrices = np.array([y_quantization_matrix_mine, color_quantization_matrix_mine])

        compressed_img = jpeg_compression(img, quantization_matrices)
        decompressed_img = jpeg_decompression(compressed_img, img.shape, quantization_matrices)
        axes[i // 3, i % 3].imshow(decompressed_img)
        axes[i // 3, i % 3].set_title("Quality Factor: {}".format(p))

    fig.savefig("jpeg_visualization.png")


def get_deflated_bytesize(data):
    raw_data = pickle.dumps(data)
    with io.BytesIO() as buf:
        with (
            zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zipf,
            zipf.open("data", mode="w") as handle,
        ):
            handle.write(raw_data)
            handle.flush()
            handle.close()
            zipf.close()
        buf.flush()
        return buf.getbuffer().nbytes


def compression_pipeline(img, c_type, param=1):
    """Pipeline для PCA и JPEG
    Вход: исходное изображение; название метода - 'pca', 'jpeg';
    param - кол-во компонент в случае PCA, и Quality Factor для JPEG
    Выход: изображение; количество бит на пиксель
    """

    assert c_type.lower() == "jpeg" or c_type.lower() == "pca"

    if c_type.lower() == "jpeg":
        y_quantization = own_quantization_matrix(y_quantization_matrix, param)
        color_quantization = own_quantization_matrix(color_quantization_matrix, param)
        matrixes = [y_quantization, color_quantization]

        compressed = jpeg_compression(img, matrixes)
        img = jpeg_decompression(compressed, img.shape, matrixes)
        compressed_size = get_deflated_bytesize(compressed)

    elif c_type.lower() == "pca":
        compressed = [
            pca_compression(c.copy(), param)
            for c in img.transpose(2, 0, 1).astype(np.float64)
        ]

        img = pca_decompression(compressed)
        compressed_size = sum(d.nbytes for c in compressed for d in c)

    raw_size = img.nbytes

    return img, compressed_size / raw_size


def calc_metrics(img_path, c_type, param_list):
    """Подсчет PSNR и Compression Ratio для PCA и JPEG. Построение графиков
    Вход: пусть до изображения; тип сжатия; список параметров: кол-во компонент в случае PCA, и Quality Factor для JPEG
    """

    assert c_type.lower() == "jpeg" or c_type.lower() == "pca"

    img = imread(img_path)
    if len(img.shape) == 3:
        img = img[..., :3]

    outputs = []
    for param in param_list:
        outputs.append(compression_pipeline(img.copy(), c_type, param))

    psnr = [peak_signal_noise_ratio(img, output[0]) for output in outputs]
    ratio = [output[1] for output in outputs]

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    fig.set_figwidth(20)
    fig.set_figheight(5)

    ax1.set_title("Quality Factor vs PSNR for {}".format(c_type.upper()))
    ax1.plot(param_list, psnr, "tab:orange")
    ax1.set_ylim(13, 64)
    ax1.set_xlabel("Quality Factor")
    ax1.set_ylabel("PSNR")

    ax2.set_title("PSNR vs Compression Ratio for {}".format(c_type.upper()))
    ax2.plot(psnr, ratio, "tab:red")
    ax2.set_xlim(13, 30)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("PSNR")
    ax2.set_ylabel("Compression Ratio")
    return fig


def get_pca_metrics_graph():
    plt.clf()
    fig = calc_metrics("Lenna.png", "pca", [1, 5, 10, 20, 50, 100, 150, 200, 256])
    fig.savefig("pca_metrics_graph.png")


def get_jpeg_metrics_graph():
    plt.clf()
    fig = calc_metrics("Lenna.png", "jpeg", [1, 10, 20, 50, 80, 100])
    fig.savefig("jpeg_metrics_graph.png")


if __name__ == "__main__":
    #pca_visualize()
    #get_gauss_1()
    #get_gauss_2()
    jpeg_visualize()
    #get_pca_metrics_graph()
    #get_jpeg_metrics_graph()
    