import numpy as np

from nn import Module


class Linear(Module):
    def __init__(self, input_channel, output_channel):
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.params = {'weight': None, 'bias': None}
        self.grads = {'weight': None, 'bias': None}

    def initialize(self, rng):
        gain = np.sqrt(2)
        fan_in = self.input_channel
        fan_out = self.output_channel
        bound = gain * np.sqrt(3 / fan_in)
        self.params['weight'] = rng.uniform(-bound, bound,
                                            (self.output_channel, self.input_channel))
        bound = 1 / np.sqrt(fan_in)
        self.params['bias'] = rng.uniform(-bound, bound, (self.output_channel,))

    def forward(self, input):
        """
        The forward pass of a linear layer.
        Store anything you need for the backward pass in self.
        Args:
            input: N x input_channel
        Returns:
            output: N x output_channel
        """
        assert (input.ndim == 2)
        assert (input.shape[1] == self.input_channel)

        self.input = input
        self.output = np.dot(input, self.params['weight'].T) + self.params['bias']
        return self.output

    def backward(self, delta):
        """
        Backward pass of a linear layer.
        Use the values stored from the forward pass to compute gradients.
        Store the gradients in `self.grads` dict.
        :param delta: Upstream gradient, N x output_channel.
        :return: downstream gradient, N x input_channel.
        """
        assert (delta.ndim == 2)
        assert (delta.shape[1] == self.output_channel)

        self.grads['weight'] = np.dot(delta.T, self.input)
        self.grads['bias'] = np.sum(delta, axis=0)
        return np.dot(delta, self.params['weight'])


class Flatten(Module):
    def __init__(self):
        self.params = {}
        self.grads = {}

    def initialize(self, rng):
        pass

    def forward(self, input):
        """
        Args:
            input: (N, any shape)
        Returns:
            output: (N, product of input shape)
        """
        self.input_shape = input.shape
        return input.reshape(input.shape[0], -1)

    def backward(self, delta):
        """
        Args:
            delta: (N, product of input shape)
        Returns:
            output: (N, any shape)
        """
        return delta.reshape(self.input_shape)


class ReLU(Module):
    def __init__(self):
        self.params = {}
        self.grads = {}

    def initialize(self, rng):
        pass

    def forward(self, input):
        """
        Args:
            input: any shape
        Returns:
            output: same shape as the input.
        """
        self.input = input
        return np.maximum(0, input)

    def backward(self, delta):
        """
        Args:
            delta: upstream gradient, any shape
        Returns:
            gradient: same shape as the input.
        """
        dx = delta * (self.input > 0)
        return dx


class Sequential(Module):
    def __init__(self, *layers):
        self.params = {}
        self.layers = list(layers)
        self.grads = {}
        self.rng = np.random.RandomState(1234)

    def set_seed(self, seed):
        self.rng = np.random.RandomState(seed)

    def initialize(self, rng):
        for layer in self.layers:
            layer.initialize(self.rng)

    def forward(self, x):
        """
        Args:
            x: input to the network
        Returns:
            output: output of the network (after the last layer)
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, delta):
        """
        Args:
            delta: gradient from the loss
        Returns:
            delta: gradient to be passed to the previous layer
        """
        for layer in reversed(self.layers):
            delta = layer.backward(delta)

        self.grads = {}
        for i, layer in enumerate(self.layers):
            for name, grad in layer.grads.items():
                self.grads[f"{i}_{name}"] = grad

        return delta



class MaxPool2d(Module):
    def __init__(self, kernel_size):
        self.params = {}
        self.grads = {}
        self.kernel_size = kernel_size

    def initialize(self, rng):
        pass

    def forward(self, input):
        """
        Pool the input by taking the max value over non-overlapping kernel_size x kernel_size blocks.
        Hint: Use a double for-loop to do this.
        Args:
            input: images of (N, H, W, C)
        Returns:
            output: pooled images of (N, H // k_h, W // k_w, C).
        """
        assert (input.ndim == 4)
        assert (input.shape[1] % self.kernel_size[0] == 0)
        assert (input.shape[2] % self.kernel_size[1] == 0)

        k_h, k_w = self.kernel_size
        N, H, W, C = input.shape
        H_out, W_out = H // k_h, W // k_w
        output = np.zeros((N, H_out, W_out, C))
        self.input = input
        self.max_mask = np.zeros_like(input, dtype=bool)

        for i in range(H_out):
            for j in range(W_out):
                window = input[:, i*k_h:(i+1)*k_h, j*k_w:(j+1)*k_w, :]
                max_val = np.max(window, axis=(1, 2), keepdims=True)
                output[:, i, j, :] = max_val.squeeze()
                self.max_mask[:, i*k_h:(i+1)*k_h, j*k_w:(j+1)*k_w, :] = (window == max_val)
        return output

    def backward(self, delta):
        """
        Args:
            delta: upstream gradient, same shape as the output
        Returns:
            gradient: same shape as the input.
        """
        N, H, W, C = self.input.shape
        k_h, k_w = self.kernel_size
        H_out, W_out = H // k_h, W // k_w
        grad_input = np.zeros_like(self.input)

        for i in range(H_out):
            for j in range(W_out):
                grad_input[:, i*k_h:(i+1)*k_h, j*k_w:(j+1)*k_w, :] += (
                    self.max_mask[:, i*k_h:(i+1)*k_h, j*k_w:(j+1)*k_w, :]
                    * delta[:, i:i+1, j:j+1, :]
                )
        return grad_input


class Conv2d(Module):
    def __init__(self, input_channel, output_channel, kernel_size):
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.kernel_size = kernel_size
        self.params = {
            'weight': np.zeros((self.output_channel,
                                self.input_channel,
                                self.kernel_size[0],
                                self.kernel_size[1])),
            'bias': np.zeros((self.output_channel,)),
        }
        self.grads = {
            'weight': np.zeros((self.output_channel,
                                self.input_channel,
                                self.kernel_size[0],
                                self.kernel_size[1])),
            'bias': np.zeros((self.output_channel,)),
        }

    def initialize(self, rng):
        gain = np.sqrt(2)
        fan_in = self.input_channel * self.kernel_size[0] * self.kernel_size[1]
        fan_out = self.output_channel
        bound = gain * np.sqrt(3 / fan_in)
        self.params['weight'] = rng.uniform(-bound, bound, self.params['weight'].shape)
        bound = 1 / np.sqrt(fan_in)
        self.params['bias'] = rng.uniform(-bound, bound, self.params['bias'].shape)

    def forward(self, input):
        """
        Convolve the input with the kernel and return the result.
        Hint:
            1. Use a double for-loop to do this.
            2. Recall the size of the kernel weight is (C_out, C_in, k_h, k_w)
                and the size of the kernel bias is (C_out,).
        Args:
            input: images of (N, H, W, C_in)
        Returns:
            output: images of (N, H', W', C_out) where H' = H - k_h + 1, W' = W - k_w + 1
        """
        self.input = input
        N, H, W, C_in = input.shape
        C_out = self.output_channel
        k_h, k_w = self.kernel_size
        H_out, W_out = H - k_h + 1, W - k_w + 1
        output = np.zeros((N, H_out, W_out, C_out))

        for i in range(H_out):
            for j in range(W_out):
                window = input[:, i:i+k_h, j:j+k_w, :]
                for c in range(C_out):
                    w = self.params['weight'][c]
                    b = self.params['bias'][c]
                    output[:, i, j, c] = np.sum(window * w.transpose(1, 2, 0), axis=(1, 2, 3)) + b
        return output

    def backward(self, delta):
        """
        Gradient with respect to the weights should be calculated and stored here.
        Args:
            delta: upstream gradient, same shape as the output
        Returns:
            gradient: same shape as the input.
        """
        N, H, W, C_in = self.input.shape
        C_out = self.output_channel
        k_h, k_w = self.kernel_size
        H_out, W_out = delta.shape[1:3]
        grad_input = np.zeros_like(self.input)
        grad_weight = np.zeros_like(self.params['weight'])
        grad_bias = np.zeros_like(self.params['bias'])

        for i in range(H_out):
            for j in range(W_out):
                window = self.input[:, i:i+k_h, j:j+k_w, :]
                for c in range(C_out):
                    d = delta[:, i, j, c][:, None, None, None]
                    grad_weight[c] += np.sum(window * d, axis=0).transpose(2, 0, 1)
                    grad_bias[c] += np.sum(delta[:, i, j, c])
                    grad_input[:, i:i+k_h, j:j+k_w, :] += d * self.params['weight'][c].transpose(1, 2, 0)

        self.grads['weight'] = grad_weight
        self.grads['bias'] = grad_bias
        return grad_input
