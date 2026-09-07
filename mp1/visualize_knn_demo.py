from models import NearestNeighbor
from utils import visualize_knn, get_mnist_dataset

X_val, y_val = get_mnist_dataset('val', size=5)

X_train_100, y_train_100 = get_mnist_dataset('train', size=100)
nn_100 = NearestNeighbor(X_train_100, y_train_100, k=10)
neighbors_100 = nn_100.get_nearest_neighbors(X_val, k=10)
visualize_knn(X_val, neighbors_100, file_name="knn_100.png")

X_train_10k, y_train_10k = get_mnist_dataset('train', size=10000)
nn_10k = NearestNeighbor(X_train_10k, y_train_10k, k=10)
neighbors_10k = nn_10k.get_nearest_neighbors(X_val, k=10)
visualize_knn(X_val, neighbors_10k, file_name="knn_10k.png")
