import numpy as np
import torch
from torch.utils.data import Dataset
from abc import ABC, abstractmethod
from sklearn.datasets import make_moons, make_circles, make_blobs


class Base2DDataset(Dataset, ABC):
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        data = (data - data.mean(dim=0)) / data.std(dim=0)  # Normalize the data
        return data

    @abstractmethod
    def generate_data(self):
        pass

class TwoMoonsDataset(Base2DDataset):
    def __init__(self, n_samples=1000, n_points=500, noise=0.1):
        data = [self.generate_data(n_points, noise) for _ in range(n_samples)]
        super().__init__(data)

    def generate_data(self, n_points, noise):
        data, _ = make_moons(n_samples=n_points, noise=noise)
        return data

class TwoCirclesDataset(Base2DDataset):
    def __init__(self, n_samples=1000, noise=0.05, factor=0.5):
        data = self.generate_data(n_samples, noise, factor)
        super().__init__(data)

    def generate_data(self, n_samples, noise, factor):
        data, _ = make_circles(n_samples=n_samples, noise=noise, factor=factor)
        return data

class GaussianBlobsDataset(Base2DDataset):
    def __init__(self, n_samples=1000, centers=3, cluster_std=1.0):
        data = self.generate_data(n_samples, centers, cluster_std)
        super().__init__(data)

    def generate_data(self, n_samples, centers, cluster_std):
        data, _ = make_blobs(n_samples=n_samples, centers=centers, cluster_std=cluster_std)
        return data

class SwissRollDataset(Base2DDataset):
    def __init__(self, n_samples=1000):
        data = self.generate_data(n_samples)
        super().__init__(data)

    def generate_data(self, n_samples):
        t = 3 * np.pi * (1 + 2 * np.random.rand(n_samples))
        x = t * np.cos(t)
        y = t * np.sin(t)
        data = np.vstack((x, y)).T
        return data

class SpiralDataset(Base2DDataset):
    def __init__(self, n_samples=1000, noise=0.5):
        data = self.generate_data(n_samples, noise)
        super().__init__(data)

    def generate_data(self, n_samples, noise):
        n = np.sqrt(np.random.rand(n_samples)) * 780 * (2 * np.pi) / 360
        d1x = -np.cos(n) * n + np.random.rand(n_samples) * noise
        d1y = np.sin(n) * n + np.random.rand(n_samples) * noise
        data = np.vstack((d1x, d1y)).T
        return data

class StarDataset(Base2DDataset):
    def __init__(self, n_samples=1000):
        data = self.generate_data(n_samples)
        super().__init__(data)

    def generate_data(self, n_samples):
        angles = np.linspace(0, 2 * np.pi, 10, endpoint=False)
        radius_outer = 1.0
        radius_inner = 0.5

        x_outer = radius_outer * np.cos(angles[::2])
        y_outer = radius_outer * np.sin(angles[::2])
        x_inner = radius_inner * np.cos(angles[1::2])
        y_inner = radius_inner * np.sin(angles[1::2])

        x = np.empty(10)
        y = np.empty(10)
        x[0::2] = x_outer
        y[0::2] = y_outer
        x[1::2] = x_inner
        y[1::2] = y_inner

        # Generate samples by interpolating along the edges of the star
        data = []
        for i in range(10):
            for _ in range(n_samples // 10):
                t = np.random.rand()
                x_point = t * x[i] + (1 - t) * x[(i + 1) % 10]
                y_point = t * y[i] + (1 - t) * y[(i + 1) % 10]
                data.append([x_point, y_point])

        data = np.array(data)
        return data

class HeartDataset(Base2DDataset):
    def __init__(self, n_samples=1000):
        data = self.generate_data(n_samples)
        super().__init__(data)

    def generate_data(self, n_samples):
        t = np.linspace(0, 2 * np.pi, n_samples)
        x = 16 * np.sin(t) ** 3
        y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
        data = np.vstack((x, y)).T
        data += np.random.randn(*data.shape) * 0.1  # Add some noise
        return data

class CircleDataset(Base2DDataset):
    def __init__(self, n_samples=1000, radius=1.0):
        data = self.generate_data(n_samples, radius)
        super().__init__(data)

    def generate_data(self, n_samples, radius):
        angles = 2 * np.pi * np.random.rand(n_samples)
        x = radius * np.cos(angles)
        y = radius * np.sin(angles)
        data = np.vstack((x, y)).T
        return data

class LineDataset(Base2DDataset):
    def __init__(self, n_samples=1000):
        data = self.generate_data(n_samples)
        super().__init__(data)

    def generate_data(self, n_samples):
        x = np.linspace(-1, 1, n_samples)
        y = 2 * x + np.random.randn(n_samples) * 0.1
        data = np.vstack((x, y)).T
        return data

class WaveDataset(Base2DDataset):
    def __init__(self, n_samples=1000, frequency=3, amplitude=1.0):
        data = self.generate_data(n_samples, frequency, amplitude)
        super().__init__(data)

    def generate_data(self, n_samples, frequency, amplitude):
        x = np.linspace(-1, 1, n_samples)
        y = amplitude * np.sin(2 * np.pi * frequency * x) + np.random.randn(n_samples) * 0.1
        data = np.vstack((x, y)).T
        return data

# If the script is run directly, visualize the datasets
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    datasets = {
        "Two Moons": TwoMoonsDataset(),
        "Two Circles": TwoCirclesDataset(),
        "Gaussian Blobs": GaussianBlobsDataset(),
        "Swiss Roll": SwissRollDataset(),
        "Spiral": SpiralDataset(),
        "Star": StarDataset(),
        "Heart": HeartDataset(),
        "Circle": CircleDataset(),
        "Line": LineDataset(),
        "Wave": WaveDataset(),
    }

    fig, axs = plt.subplots(2, 5, figsize=(20, 8))
    axs = axs.ravel()

    for i, (name, dataset) in enumerate(datasets.items()):
        data = np.array([dataset[j].numpy() for j in range(len(dataset))])
        axs[i].scatter(data[:, 0], data[:, 1], s=10)
        axs[i].set_title(name)
        axs[i].set_xticks([])
        axs[i].set_yticks([])

    plt.show()
