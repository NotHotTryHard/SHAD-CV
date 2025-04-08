from interface import *


# ================================= 1.4.1 SGD ================================
class SGD(Optimizer):
    def __init__(self, lr):
        self.lr = lr

    def get_parameter_updater(self, parameter_shape):
        """
        :param parameter_shape: tuple, the shape of the associated parameter

        :return: the updater function for that parameter
        """

        def updater(parameter, parameter_grad):
            """
            :param parameter: np.array, current parameter values
            :param parameter_grad: np.array, current gradient, dLoss/dParam

            :return: np.array, new parameter values
            """
            # your code here \/
            return parameter - self.lr * parameter_grad
            # your code here /\

        return updater


# ============================= 1.4.2 SGDMomentum ============================
class SGDMomentum(Optimizer):
    def __init__(self, lr, momentum=0.0):
        self.lr = lr
        self.momentum = momentum

    def get_parameter_updater(self, parameter_shape):
        """
        :param parameter_shape: tuple, the shape of the associated parameter

        :return: the updater function for that parameter
        """

        def updater(parameter, parameter_grad):
            """
            :param parameter: np.array, current parameter values
            :param parameter_grad: np.array, current gradient, dLoss/dParam

            :return: np.array, new parameter values
            """
            # your code here \/
            updater.inertia = self.momentum * updater.inertia + self.lr * parameter_grad
            return parameter - updater.inertia
            # your code here /\

        updater.inertia = np.zeros(parameter_shape)
        return updater



# ================================ 2.1.1 ReLU ================================
class ReLU(Layer):
    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, ...)), input values

        :return: np.array((n, ...)), output values

            n - batch size
            ... - arbitrary shape (the same for input and output)
        """
        # your code here \/
        return np.maximum(inputs, 0)
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, ...)), dLoss/dOutputs

        :return: np.array((n, ...)), dLoss/dInputs

            n - batch size
            ... - arbitrary shape (the same for input and output)
        """
        # your code here \/
        return grad_outputs * (self.forward_inputs >= 0)
        # your code here /\


# =============================== 2.1.2 Softmax ==============================
class Softmax(Layer):
    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, d)), input values

        :return: np.array((n, d)), output values

            n - batch size
            d - number of units
        """
        # your code here \/
        exp_inputs = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        sigma_z = exp_inputs / np.sum(exp_inputs, axis=1, keepdims=True)
        
        return sigma_z
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, d)), dLoss/dOutputs

        :return: np.array((n, d)), dLoss/dInputs

            n - batch size
            d - number of units
        """
        # your code here \/
        sigma_z = self.forward_outputs

        return (grad_outputs - np.sum(grad_outputs * sigma_z, axis=1, keepdims=True)) * sigma_z
        # your code here /\


# ================================ 2.1.3 Dense ===============================
class Dense(Layer):
    def __init__(self, units, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_units = units

        self.weights, self.weights_grad = None, None
        self.biases, self.biases_grad = None, None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        (input_units,) = self.input_shape
        output_units = self.output_units

        # Register weights and biases as trainable parameters
        # Note, that the parameters and gradients *must* be stored in
        # self.<p> and self.<p>_grad, where <p> is the name specified in
        # self.add_parameter

        self.weights, self.weights_grad = self.add_parameter(
            name="weights",
            shape=(output_units, input_units),
            initializer=he_initializer(input_units),
        )

        self.biases, self.biases_grad = self.add_parameter(
            name="biases",
            shape=(output_units,),
            initializer=np.zeros,
        )

        self.output_shape = (output_units,)

    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, d)), input values

        :return: np.array((n, c)), output values

            n - batch size
            d - number of input units
            c - number of output units
        """
        # your code here \/
        return inputs @ self.weights.T + self.biases
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, c)), dLoss/dOutputs

        :return: np.array((n, d)), dLoss/dInputs

            n - batch size
            d - number of input units
            c - number of output units
        """
        # your code here \/
        self.biases_grad = np.sum(grad_outputs, axis=0)
        self.weights_grad = (grad_outputs.T @ self.forward_inputs)
        return (self.weights.T @ grad_outputs.T).T
        # your code here /\


# ============================ 2.2.1 Crossentropy ============================
class CategoricalCrossentropy(Loss):
    def value_impl(self, y_gt, y_pred):
        """
        :param y_gt: np.array((n, d)), ground truth (correct) labels
        :param y_pred: np.array((n, d)), estimated target values

        :return: np.array((1,)), mean Loss scalar for batch

            n - batch size
            d - number of units
        """
        # your code here \/
        y_pred_clip = np.clip(y_pred, eps, 1 - eps)
        return np.mean(-np.sum(np.log(y_pred_clip) * y_gt, axis=1), axis=0, keepdims=True)
        # your code here /\

    def gradient_impl(self, y_gt, y_pred):
        """
        :param y_gt: np.array((n, d)), ground truth (correct) labels
        :param y_pred: np.array((n, d)), estimated target values

        :return: np.array((n, d)), dLoss/dY_pred

            n - batch size
            d - number of units
        """
        # your code here \/
        y_pred_clip = np.clip(y_pred, eps, 1 - eps)
        return -y_gt / y_pred_clip / y_pred.shape[0]
        # your code here /\


# ======================== 2.3 Train and Test on MNIST =======================
def train_mnist_model(x_train, y_train, x_valid, y_valid):
    # your code here \/
    # 1) Create a Model
    model = Model(CategoricalCrossentropy(), SGDMomentum(lr=0.05, momentum=0.4))

    # 2) Add layers to the model
    #   (don't forget to specify the input shape for the first layer)
    model.add(Dense(128, (784, )))
    model.add(ReLU())
    model.add(Dense(64))
    model.add(ReLU())
    model.add(Dense(10))
    model.add(Softmax())

    print(model)

    # 3) Train and validate the model using the provided data
    model.fit(x_train, y_train, 64, 12, x_valid=x_valid, y_valid=y_valid)

    # your code here /\
    return model


# ============================== 3.3.2 convolve ==============================
def convolve(inputs, kernels, padding=0):
    """
    :param inputs: np.array((n, d, ih, iw)), input values
    :param kernels: np.array((c, d, kh, kw)), convolution kernels
    :param padding: int >= 0, the size of padding, 0 means 'valid'

    :return: np.array((n, c, oh, ow)), output values

        n - batch size
        d - number of input channels
        c - number of output channels
        (ih, iw) - input image shape
        (oh, ow) - output image shape
    """
    # !!! Don't change this function, it's here for your reference only !!!
    assert isinstance(padding, int) and padding >= 0
    assert inputs.ndim == 4 and kernels.ndim == 4
    assert inputs.shape[1] == kernels.shape[1]

    if os.environ.get("USE_FAST_CONVOLVE", False):
        return convolve_pytorch(inputs, kernels, padding)
    else:
        return convolve_numpy(inputs, kernels, padding)


def convolve_numpy(inputs, kernels, padding):
    """
    :param inputs: np.array((n, d, ih, iw)), input values
    :param kernels: np.array((c, d, kh, kw)), convolution kernels
    :param padding: int >= 0, the size of padding, 0 means 'valid'

    :return: np.array((n, c, oh, ow)), output values

        n - batch size
        d - number of input channels
        c - number of output channels
        (ih, iw) - input image shape
        (oh, ow) - output image shape
    """
    n, c = inputs.shape[0], kernels.shape[0]
    Tf = np.array(inputs.shape[2:])
    Tg = np.array(kernels.shape[2:])
    To = Tf - Tg + 1 + 2 * padding
    
    if padding > 0:
        inputs_padded = np.pad(inputs, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')
    else:
        inputs_padded = inputs
    
    outputs = np.zeros((n, c, To[0], To[1]))
    
    for i in range(n):      # по каждому изображению в батче
        for j in range(c):  # по каждому каналу выхода
            # Вырезаем все позиции фильтра для одного канала
            for y in range(To[0]):
                for x in range(To[1]):
                    # Извлекаем область с нужными размерами
                    region = inputs_padded[i, :, y:y + Tg[0], x:x + Tg[1]][:, ::-1, ::-1]
                    # Применяем ядро, умножая и суммируя
                    outputs[i, j, y, x] = np.sum(region * kernels[j])
                    
    
    return outputs

# =============================== 4.1.1 Conv2D ===============================
class Conv2D(Layer):
    def __init__(self, output_channels, kernel_size=3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert kernel_size % 2, "Kernel size should be odd"

        self.output_channels = output_channels
        self.kernel_size = kernel_size

        self.kernels, self.kernels_grad = None, None
        self.biases, self.biases_grad = None, None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        input_channels, input_h, input_w = self.input_shape
        output_channels = self.output_channels
        kernel_size = self.kernel_size

        self.kernels, self.kernels_grad = self.add_parameter(
            name="kernels",
            shape=(output_channels, input_channels, kernel_size, kernel_size),
            initializer=he_initializer(input_h * input_w * input_channels),
        )

        self.biases, self.biases_grad = self.add_parameter(
            name="biases",
            shape=(output_channels,),
            initializer=np.zeros,
        )

        self.output_shape = (output_channels,) + self.input_shape[1:]

    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, d, h, w)), input values

        :return: np.array((n, c, h, w)), output values

            n - batch size
            d - number of input channels
            c - number of output channels
            (h, w) - image shape
        """
        # your code here \/
        tmp = convolve(inputs, self.kernels, padding=(self.kernel_size - 1) // 2) 
        outputs = tmp + self.biases.reshape(1, tmp.shape[1], 1, 1)
        return outputs
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, c, h, w)), dLoss/dOutputs

        :return: np.array((n, d, h, w)), dLoss/dInputs

            n - batch size
            d - number of input channels
            c - number of output channels
            (h, w) - image shape
        """
        # your code here \/
        p = (self.kernel_size - 1) // 2
        self.kernels_grad = np.transpose(convolve(np.transpose(self.forward_inputs, (1, 0, 2, 3))[:, :, ::-1, ::-1], np.transpose(grad_outputs, (1, 0, 2, 3)), p), (1, 0, 2, 3))
        self.biases_grad = np.sum(grad_outputs, axis=(0, 2, 3))
        
        return convolve(grad_outputs, np.transpose(self.kernels, (1, 0, 2, 3))[:, :, ::-1, ::-1], p)
        # your code here /\


# ============================== 4.1.2 Pooling2D =============================
class Pooling2D(Layer):
    def __init__(self, pool_size=2, pool_mode="max", *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert pool_mode in {"avg", "max"}

        self.pool_size = pool_size
        self.pool_mode = pool_mode
        self.forward_idxs = None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        channels, input_h, input_w = self.input_shape
        output_h, rem_h = divmod(input_h, self.pool_size)
        output_w, rem_w = divmod(input_w, self.pool_size)
        assert not rem_h, "Input height should be divisible by the pool size"
        assert not rem_w, "Input width should be divisible by the pool size"

        self.output_shape = (channels, output_h, output_w)

    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, d, ih, iw)), input values

        :return: np.array((n, d, oh, ow)), output values

            n - batch size
            d - number of channels
            (ih, iw) - input image shape
            (oh, ow) - output image shape
        """
        # your code here \/
        (n, d, ih, iw) = inputs.shape
        (_, oh, ow) = self.output_shape
        
        inputs_reshaped = inputs.reshape(n, d, oh, self.pool_size, ow, self.pool_size)
        inputs_reshaped = inputs_reshaped.transpose(0, 1, 2, 4, 3, 5)
        #print(inputs_reshaped)
        inputs_reshaped = inputs_reshaped.reshape(n, d, oh, ow, -1)
        #print(inputs_reshaped)
        
        if self.pool_mode == 'max':
            self.maxidx = np.argmax(inputs_reshaped, axis=(4))
            outputs = np.max(inputs_reshaped, axis=(4))
        elif self.pool_mode == 'avg':
            outputs = np.mean(inputs_reshaped, axis=(4))
        else:
            outputs = None
        return outputs
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, d, oh, ow)), dLoss/dOutputs

        :return: np.array((n, d, ih, iw)), dLoss/dInputs

            n - batch size
            d - number of channels
            (ih, iw) - input image shape
            (oh, ow) - output image shape
        """
        # your code here \/
        (n, d, oh, ow) = grad_outputs.shape
        if self.pool_mode == 'avg':
            outputs = np.repeat(grad_outputs[:, :, np.newaxis] / self.pool_size ** 2, self.pool_size ** 2, axis=-1)
        elif self.pool_mode == 'max':
            outputs = np.repeat(np.zeros_like(grad_outputs)[:, :, np.newaxis], self.pool_size ** 2, axis=-1).reshape((n, d, oh, ow, self.pool_size ** 2))
            index_lists = [np.arange(s) for s in outputs.shape[:-1]]
            grid_indices = np.ix_(*index_lists)
            outputs[grid_indices + (self.maxidx,)] = grad_outputs
        outputs = outputs.reshape((n, d, oh, ow, self.pool_size, self.pool_size)).transpose(0, 1, 2, 4, 3, 5).reshape(self.forward_inputs.shape)
        return outputs
        # your code here /\
'''(n, d, oh, ow) = grad_outputs.shape
            outputs = np.repeat(grad_outputs[:, :, np.newaxis] / self.pool_size ** 2, self.pool_size ** 2, axis=-1)
            outputs = outputs.reshape((n, d, oh, ow, self.pool_size, self.pool_size))
            outputs = outputs.transpose(0, 1, 2, 4, 3, 5)
            outputs = outputs.reshape(self.forward_inputs.shape)'''
'''(n, d, oh, ow) = grad_outputs.shape
            print(self.pool_size)
            print(grad_outputs)
            print(self.maxidx)
            outputs = np.repeat(np.zeros_like(grad_outputs)[:, :, np.newaxis], self.pool_size ** 2, axis=-1)
            outputs = outputs.reshape((n, d, oh, ow, self.pool_size ** 2))
            index_lists = [np.arange(s) for s in outputs.shape[:-1]]
            grid_indices = np.ix_(*index_lists)
            outputs[grid_indices + (self.maxidx,)] = grad_outputs
            outputs = outputs.reshape((n, d, oh, ow, self.pool_size, self.pool_size))
            outputs = outputs.transpose(0, 1, 2, 4, 3, 5)
            outputs = outputs.reshape(self.forward_inputs.shape)'''

# ============================== 4.1.3 BatchNorm =============================
class BatchNorm(Layer):
    def __init__(self, momentum=0.9, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.momentum = momentum

        self.running_mean = None
        self.running_var = None

        self.beta, self.beta_grad = None, None
        self.gamma, self.gamma_grad = None, None

        self.forward_inverse_std = None
        self.forward_centered_inputs = None
        self.forward_normalized_inputs = None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        input_channels, input_h, input_w = self.input_shape
        self.running_mean = np.zeros((input_channels,))
        self.running_var = np.ones((input_channels,))

        self.beta, self.beta_grad = self.add_parameter(
            name="beta",
            shape=(input_channels,),
            initializer=np.zeros,
        )

        self.gamma, self.gamma_grad = self.add_parameter(
            name="gamma",
            shape=(input_channels,),
            initializer=np.ones,
        )

    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, d, h, w)), input values

        :return: np.array((n, d, h, w)), output values

            n - batch size
            d - number of channels
            (h, w) - image shape
        """
        # your code here \/
        if self.is_training:
            mean = np.mean(inputs, axis=(0, 2, 3))
            var = np.var(inputs, axis=(0, 2, 3))
            a = np.transpose(inputs, (0, 3, 2, 1)) - mean
            self.forward_centered_inputs = np.transpose(a, (0, 3, 2, 1))
            self.forward_inverse_std = 1 / np.sqrt(eps + var)
            outputs = a * self.forward_inverse_std
            self.forward_normalized_inputs = np.transpose(outputs, (0, 3, 2, 1))
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            outputs = (np.transpose(inputs, (0, 3, 2, 1)) - self.running_mean) / np.sqrt(eps + self.running_var)
        outputs = np.transpose(self.gamma * outputs + self.beta, (0, 3, 2, 1))
        return outputs
        # your code here /\
    
    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, d, h, w)), dLoss/dOutputs

        :return: np.array((n, d, h, w)), dLoss/dInputs

            n - batch size
            d - number of channels
            (h, w) - image shape
        """
        # your code here \/
        #n, d, h, w = grad_outputs.shape

        # Градиенты по gamma и beta
        #self.gamma_grad = np.sum(grad_outputs * self.forward_normalized_inputs, axis=(0, 2, 3))
        #self.beta_grad = np.sum(grad_outputs, axis=(0, 2, 3))

        # Градиенты по нормализованным данным
        #

        # Градиенты по дисперсии и среднему
        #grad_var = np.sum(grad_normalized * self.forward_centered_inputs, axis=(0, 2, 3), keepdims=True) * -0.5 * (self.forward_inverse_std**3)
        #grad_mean = np.sum(grad_normalized * -self.forward_inverse_std, axis=(0, 2, 3), keepdims=True) + grad_var * np.mean(-2 * self.forward_centered_inputs, axis=(0, 2, 3), keepdims=True)

        # Градиенты по входным данным
        #grad_inputs = grad_normalized * self.forward_inverse_std + grad_var * 2 * self.forward_centered_inputs / (n * h * w) + grad_mean / (n * h * w)

        n, d, h, w = grad_outputs.shape

        self.gamma_grad = np.sum(grad_outputs * self.forward_normalized_inputs, axis=(0, 2, 3))
        self.beta_grad = np.sum(grad_outputs, axis=(0, 2, 3))
        

        grad_normalized = grad_outputs * self.gamma[np.newaxis, :, np.newaxis, np.newaxis]
        grad_var = np.sum(grad_normalized * self.forward_centered_inputs * (-0.5) * self.forward_inverse_std[:, np.newaxis, np.newaxis] ** 3, axis=(0, 2, 3))        
        grad_mean = np.sum(grad_normalized * (-self.forward_inverse_std[:, np.newaxis, np.newaxis]), axis=(0, 2, 3), keepdims=False)[:, np.newaxis, np.newaxis] + grad_var[:, np.newaxis, np.newaxis] * np.sum(-2 * self.forward_centered_inputs, axis=(0, 2, 3), keepdims=False)[:, np.newaxis, np.newaxis] / (n * h * w) 
        return grad_normalized * self.forward_inverse_std[:, np.newaxis, np.newaxis] + (grad_var[:, np.newaxis, np.newaxis] * 2 * self.forward_centered_inputs + grad_mean) / (n * h * w)
        # your code here /\
    
# =============================== 4.1.4 Flatten ==============================
class Flatten(Layer):
    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        self.output_shape = (int(np.prod(self.input_shape)),)

    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, d, h, w)), input values

        :return: np.array((n, (d * h * w))), output values

            n - batch size
            d - number of input channels
            (h, w) - image shape
        """
        # your code here \/
        return inputs.reshape(inputs.shape[0], -1)
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, (d * h * w))), dLoss/dOutputs

        :return: np.array((n, d, h, w)), dLoss/dInputs

            n - batch size
            d - number of units
            (h, w) - input image shape
        """
        # your code here \/
        return grad_outputs.reshape(self.forward_inputs.shape)
        # your code here /\


# =============================== 4.1.5 Dropout ==============================
class Dropout(Layer):
    def __init__(self, p, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p = p
        self.forward_mask = None

    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, ...)), input values

        :return: np.array((n, ...)), output values

            n - batch size
            ... - arbitrary shape (the same for input and output)
        """
        # your code here \/
        if self.is_training:
            self.forward_mask = np.where(np.random.uniform(size=inputs.shape) < self.p, 0, 1)
        else:
            self.forward_mask = np.ones(inputs.shape) * (1 - self.p)
        return self.forward_mask * inputs
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, ...)), dLoss/dOutputs

        :return: np.array((n, ...)), dLoss/dInputs

            n - batch size
            ... - arbitrary shape (the same for input and output)
        """
        # your code here \/
        return grad_outputs * self.forward_mask
        # your code here /\


# ====================== 2.3 Train and Test on CIFAR-10 ======================
def train_cifar10_model(x_train, y_train, x_valid, y_valid):
    # your code here \/
    # 1) Create a Model
    model = Model(CategoricalCrossentropy(), SGDMomentum(lr=0.05, momentum=0.0))

    # 2) Add layers to the model
    #   (don't forget to specify the input shape for the first layer)
    model.add(Conv2D(16, input_shape=(3, 32, 32)))
    model.add(BatchNorm())
    model.add(ReLU())
    model.add(Pooling2D(2, 'max'))
    
    model.add(Conv2D(32))
    model.add(BatchNorm())
    model.add(ReLU())
    model.add(Pooling2D(2, 'max'))
    
    model.add(Conv2D(64))
    model.add(BatchNorm())
    model.add(ReLU())
    model.add(Pooling2D(2, 'max'))
    
    model.add(Flatten())
    model.add(Dense(128))
    model.add(ReLU())
    model.add(Dropout(0.5))

    model.add(Dense(10))
    model.add(Softmax())
    print(model)

    # 3) Train and validate the model using the provided data
    model.fit(x_train, y_train, 4, 14, x_valid=x_valid, y_valid=y_valid)

    # your code here /\
    return model


# ============================================================================