from typing import List, Tuple

import numpy as np
import torch


# Task 1 (1 point)
class QuantizationParameters:
    def __init__(
        self,
        scale: np.float64,
        zero_point: np.int32,
        q_min: np.int32,
        q_max: np.int32,
    ):
        self.scale = scale
        self.zero_point = zero_point
        self.q_min = q_min
        self.q_max = q_max

    def __repr__(self):
        return f"scale: {self.scale}, zero_point: {self.zero_point}"


def compute_quantization_params(
    r_min: np.float64,
    r_max: np.float64,
    q_min: np.int32,
    q_max: np.int32,
) -> QuantizationParameters:
    # your code goes here \/
    s = np.float64((r_max - r_min) / (q_max - q_min))
    z = np.int32(round((r_max * q_min - r_min * q_max) / (r_max - r_min)))
    return QuantizationParameters(s, z, q_min, q_max)
    # your code goes here /\


# Task 2 (0.5 + 0.5 = 1 point)
def quantize(r: np.ndarray, qp: QuantizationParameters) -> np.ndarray:
    # your code goes here \/
    return np.clip(np.round(r / qp.scale + qp.zero_point), qp.q_min, qp.q_max).astype(np.int8)
    # your code goes here /\


def dequantize(q: np.ndarray, qp: QuantizationParameters) -> np.ndarray:
    # your code goes here \/
    return (q.astype(np.int32) - qp.zero_point) * qp.scale
    # your code goes here /\


# Task 3 (1 point)
class MinMaxObserver:
    def __init__(self):
        self.min = np.finfo(np.float64).max
        self.max = np.finfo(np.float64).min

    def __call__(self, x: torch.Tensor):
        # your code goes here \/
        self.min = np.float64(min(self.min, torch.min(x).item()))
        self.max = np.float64(max(self.max, torch.max(x).item()))
        # your code goes here /\


# Task 4 (1 + 1 = 2 points)
def quantize_weights_per_tensor(
    weights: np.ndarray,
) -> Tuple[np.array, QuantizationParameters]:
    # your code goes here \/
    r_abs = np.max(np.abs(weights))
    qp = compute_quantization_params(-r_abs, r_abs, -127, 127)
    return quantize(weights, qp), qp
    # your code goes here /\


def quantize_weights_per_channel(
    weights: np.ndarray,
) -> Tuple[np.array, List[QuantizationParameters]]:
    # your code goes here \/
    res = weights.copy().astype(np.int8)
    qps = []
    for ch in range(weights.shape[0]):
        weights_tmp = weights[ch]
        r_abs = np.max(np.abs(weights_tmp))
        qp = compute_quantization_params(-r_abs, r_abs, -127, 127)
        qps.append(qp)
        res[ch] = quantize(weights_tmp, qp)
    return res, qps
    # your code goes here /\


# Task 5 (1 point)
def quantize_bias(
    bias: np.float64,
    scale_w: np.float64,
    scale_x: np.float64,
) -> np.int32:
    # your code goes here \/
    return np.int32(round(bias / (scale_w * scale_x)))
    # your code goes here /\


# Task 6 (2 points)
def quantize_multiplier(m: np.float64) -> [np.int32, np.int32]:
    # your code goes here \/
    n = 0
    while m >= 1:
        m /= 2
        n -= 1
    while m < 0.5:
        m *= 2
        n += 1
        
    M0 = round(m * 2**31)
    return np.int32(n), np.int32(M0)
    # your code goes here /\


def quantize_multiplier(m: np.float64) -> [np.int32, np.int32]:
    n = -np.ceil(np.log2(m))
    m_scaled = m / (2.0 ** (-n))

    while m_scaled >= 1:
        m_scaled /= 2
        n -= 1
    while m_scaled < 0.5:
        m_scaled *= 2
        n += 1

    M0 = round(m_scaled * (2 ** 31))
    return np.int32(n), np.int32(M0)



# Task 7 (2 points)
def multiply_by_quantized_multiplier(
    accum: np.int32,
    n: np.int32,
    m0: np.int32,
) -> np.int32:
    # your code goes here \/
    tmp = np.multiply(accum, m0, dtype=np.int64)
    shifted = tmp >> (31 + n)
    rounding_bit = (tmp >> (30 + n)) & 1

    if rounding_bit:
        shifted += 1

    #result = np.clip(shifted, np.iinfo(np.int32).min, np.iinfo(np.int32).max)

    return np.int32(shifted)
    # your code goes here /\
