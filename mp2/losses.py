import numpy as np
from nn import Module


class L2Loss(Module):
    def __init__(self):
        pass

    def initialize(self, rng):
        pass

    def forward(self, input, target) -> np.float32:
        self.input = input
        self.target = target
        diff = input.reshape(input.shape[0], -1) - target.reshape(target.shape[0], -1)
        output = np.sum(diff ** 2, axis=1)
        self.n = output.shape[0]
        output = np.sum(output) / self.n
        return output

    def backward(self, delta):
        return 2 * (self.input - self.target) * delta / self.n


class SoftmaxWithLogitsLoss(Module):
    def __init__(self):
        pass

    def initialize(self, rng):
        pass

    def forward(self, input, target) -> np.float32:
        """
        Forward pass of the softmax cross-entropy loss.
        Args:
            input: n x n_class logits
            target: n x n_class one-hot ground truth
        Returns:
            loss: scalar, mean cross-entropy
        """
        logits = input - np.max(input, axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        softmax = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        log_likelihood = -np.sum(target * np.log(softmax + 1e-12), axis=1)
        loss = np.mean(log_likelihood)

        self.softmax = softmax
        self.target = target
        self.n = input.shape[0]

        return loss.astype(np.float32)

    def backward(self, delta):
        """
        Backward pass of softmax cross-entropy loss.
        Returns gradient wrt logits.
        """
        dx = (self.softmax - self.target) / self.n
        dx *= delta
        return dx
