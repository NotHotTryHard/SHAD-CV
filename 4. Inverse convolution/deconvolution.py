import numpy as np
from scipy.fft import fft2, ifft2, ifftshift



def gaussian_kernel(size, sigma):
    """
    Построение ядра фильтра Гаусса.

    @param  size  int    размер фильтра
    @param  sigma float  параметр размытия
    @return numpy array  фильтр Гаусса размером size x size
    """
    if size % 2 == 0:
        ax = np.arange(-0.5 - size / 2 + 1, +0.5 + size / 2 - 1 + 1)
    else:
        ax = np.arange(-int(size / 2), int(size / 2) + 1)
    x_mesh, y_mesh = np.meshgrid(ax, ax)
    
    kernel = np.exp(-(x_mesh ** 2 + y_mesh ** 2) / (2. * sigma**2))
    kernel /= np.sum(kernel)
    
    return kernel


def pad_kernel(kernel, target):
    th, tw = target
    kh, kw = kernel.shape[:2]
    ph, pw = th - kh, tw - kw

    padding = [((ph + 1) // 2, ph // 2), ((pw + 1) // 2, pw // 2)]
    kernel = np.pad(kernel, padding)

    assert kernel.shape[:2] == target
    return kernel


def fourier_transform(h, shape):
    """
    Получение Фурье-образа искажающей функции

    @param  h            numpy array  искажающая функция h (ядро свертки)
    @param  shape        list         требуемый размер образа
    @return numpy array  H            Фурье-образ искажающей функции h
    """
    padded = pad_kernel(h, shape)

    assert padded.shape == shape
    return np.fft.fft2(np.fft.ifftshift(padded))
    
    
    


def inverse_kernel(H, threshold=1e-10):
    """
    Получение H_inv

    @param  H            numpy array    Фурье-образ искажающей функции h
    @param  threshold    float          порог отсечения для избежания деления на 0
    @return numpy array  H_inv
    """
    H_inv = np.zeros_like(H)
    H_inv[abs(H) > threshold] = 1 / H[abs(H) > threshold]
    return H_inv


def inverse_filtering(blurred_img, h, threshold=1e-10):
    """
    Метод инверсной фильтрации

    @param  blurred_img    numpy array  искаженное изображение
    @param  h              numpy array  искажающая функция
    @param  threshold      float        параметр получения H_inv
    @return numpy array                 восстановленное изображение
    """
    H = fourier_transform(h, blurred_img.shape)
    H_inv = inverse_kernel(H, threshold)
    
    F = np.fft.fft2(blurred_img) * H_inv
    recreated_img = np.abs(np.fft.ifft2(F))
    return recreated_img


def wiener_filtering(blurred_img, h, K=5e-5):
    """
    Винеровская фильтрация

    @param  blurred_img    numpy array  искаженное изображение
    @param  h              numpy array  искажающая функция
    @param  K              float        константа из выражения (8)
    @return numpy array                 восстановленное изображение
    """
    H = fourier_transform(h, blurred_img.shape)
    H_conj = H.conj()
    G = np.fft.fft2(blurred_img)
    
    F = (H_conj / (H * H_conj + K)) * G
    recreated_img = np.abs(np.fft.ifft2(F)).clip(0, 255)
    return recreated_img


def compute_psnr(img1, img2):
    """
    PSNR metric

    @param  img1    numpy array   оригинальное изображение
    @param  img2    numpy array   искаженное изображение
    @return float   PSNR(img1, img2)
    """
    assert type(img1) == type(img2)
    
    return 20 * np.log10(255 / np.sqrt(np.mean((img1 - img2) ** 2)))
