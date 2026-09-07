import numpy as np
import logging
from utils import compute_accuracy

class NearestNeighbor(object):
    def __init__(self, data, labels, k):
        """
        Args:
            data: n_train x d matrix with a d-dimensional feature for each of
            the n_train points
            labels: n_train vector with the label for each of the n points
            k: number of nearest neighbors to use for prediction
        """
        self.k = k
        self.Xtr = data
        self.ytr = labels.astype(int)
        self.num_classes = int(np.max(labels)) + 1

    def train(self):
        """
        Trains the model and stores in class variables whatever is necessary to
        make predictions later.
        """
        # BEGIN YOUR CODE
        pass
        # END YOUR CODE

    def predict(self, x):
        """
        Args:
            x: n_test x d matrix with a d-dimensional feature for each of the
            n_test points
        Returns:
            y: n_test vector with the predicted label for each of the n_test points
        """
        # BEGIN YOUR CODE
        num_test = x.shape[0]
        Ypred = np.zeros(num_test, dtype=self.ytr.dtype)

        for i in range(num_test):
            # L2 distance
            distances = np.sum((self.Xtr - x[i, :]) ** 2, axis=1)
            idx = np.argpartition(distances, self.k)[:self.k]
            idx = idx[np.argsort(distances[idx])]
            nearest_labels = self.ytr[idx]
            votes = np.bincount(nearest_labels)
            Ypred[i] = np.argmax(votes)
        return Ypred
        # END YOUR CODE
    
    def get_nearest_neighbors(self, x, k):
        """
        Args:
            x: n x d matrix with a d-dimensional feature for each of the n
            points
            k: number of nearest neighbors to return
        Returns:
            top_imgs: n x k x d vector containing the nearest neighbors in the
            training data, top_imgs must be sorted by the distance to the
            corresponding point in x.
        """
        # BEGIN YOUR CODE
        num_test = x.shape[0]
        D = self.Xtr.shape[1]
        top_imgs = np.zeros((num_test, k, D))

        for i in range(num_test):
            distances = np.sum((self.Xtr - x[i, :]) ** 2, axis=1)
            # get indices of k smallest distances
            idx = np.argpartition(distances, k)[:k]
            idx = idx[np.argsort(distances[idx])]
            top_imgs[i] = self.Xtr[idx]

        return top_imgs
        # END YOUR CODE





class LinearClassifier(object):
    def __init__(self, data, labels, epochs=10, lr=1e-3, reg_wt=3e-5, writer=None):
        self.data = data
        self.labels = labels
        self.epochs = epochs
        self.lr = lr
        self.reg_wt = reg_wt
        self.rng = np.random.RandomState(1234)
        std = 1. / np.sqrt(data.shape[1])
        self.w = self.rng.uniform(-std, std, size=(self.data.shape[1], 10))
        self.writer = writer

    def compute_loss_and_gradient(self):
        """
        Computes total loss and gradient of total loss with respect to weights
        w.  You may want to use the `data, w, labels, reg_wt` attributes in
        this function.
        
        Returns:
            data_loss, reg_loss, total_loss: 3 scalars that represent the
                losses $L_d$, $L_r$ and $L$ as defined in the README.
            grad_w: d x 10. The gradient of the total loss (including
            the regularization term), wrt the weight.
        """
        # BEGIN YOUR CODE
        N = self.data.shape[0]
        scores = self.data @ self.w
        scores -= np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        correct_class_log_probs = -np.log(probs[np.arange(N), self.labels])
        data_loss = np.mean(correct_class_log_probs)
        reg_loss = 0.5 * self.reg_wt * np.sum(self.w * self.w)
        total_loss = data_loss + reg_loss
        probs[np.arange(N), self.labels] -= 1
        probs /= N
        grad_w = self.data.T @ probs + self.reg_wt * self.w
        return data_loss, reg_loss, total_loss, grad_w
        # END YOUR CODE

    def train(self):
        """Train the linear classifier using gradient descent"""
        for i in range(self.epochs):
            # BEGIN YOUR CODE
            # You may want to call the `compute_loss_and_gradient` method.
            # You can also print the total loss and accuracy on the training
            # data here for debugging.
            data_loss, reg_loss, total_loss, grad_w = self.compute_loss_and_gradient()
            self.w -= self.lr * grad_w
            if self.writer is not None:
                self.writer.add_scalar("Loss/Total", total_loss, i)
                self.writer.add_scalar("Loss/Data", data_loss, i)
                self.writer.add_scalar("Loss/Reg", reg_loss, i)
            # END YOUR CODE

    def predict(self, x):
        """
        Args:
            x: n_test x d matrix with a d-dimensional feature for each of the
            n_test points
        Returns:
            y: n_test vector with the predicted label for each of the n_test
            points
        """
        scores = x @ self.w
        return np.argmax(scores, axis=1)
