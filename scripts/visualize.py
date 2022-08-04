import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
from PIL import Image


def visualize():
    for i in range(5000):
        file_name = f"./results/opinion_1_{i}.npz"
        try:
            file_dict = np.load(file_name)
            opinions, dynamic_matrix = file_dict["opinions"], file_dict["dynamic_matrix"]
            visualize_opinions(opinions, i)
        except OSError:
            print(f"Read {i} files")
            return


def visualize_opinions(opinions, index):
    # Save opinions into a file
    dimensions = opinions.shape[0]
    if dimensions == 2:
        visualize_2d(opinions, index)
    elif dimensions == 3:
        visualize_3d(opinions, index)
    else:
        raise TypeError(f"No support for visualizing opinions with dimension: {dimensions}")


def visualize_2d(opinions, index):
    num_pairs = opinions.shape[1] // 2
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    castor = [opinions[0][:num_pairs], opinions[1][:num_pairs]]
    pollux = [opinions[0][num_pairs:], opinions[1][num_pairs:]]
    colors = iter(plt.cm.rainbow(np.linspace(0, 1, num_pairs)))
    for i in range(num_pairs):
        x = [castor[0][i], pollux[0][i]]
        y = [castor[1][i], pollux[1][i]]
        color = next(colors)
        plt.scatter(castor[0][i], castor[1][i], color=color, marker='x', zorder=5)
        plt.scatter(pollux[0][i], pollux[1][i], color=color, marker='o', zorder=5)
        plt.plot(x, y, color=color, zorder=1, linestyle='--', alpha=0.5, label=f"Pair {i}")
    plt.legend(bbox_to_anchor=(1.25, 1))
    plt.title(f"{index}")
    plt.savefig(f"./figures/fig_{str(index).zfill(5)}.png", bbox_inches='tight')
    plt.close()


def visualize_3d(opinions, index):
    num_pairs = opinions.shape[1] // 2
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection='3d')
    castor = [opinions[0][:num_pairs], opinions[1][:num_pairs], opinions[2][:num_pairs]]
    pollux = [opinions[0][num_pairs:], opinions[1][num_pairs:], opinions[2][num_pairs:]]
    colors = iter(plt.cm.rainbow(np.linspace(0, 1, num_pairs)))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    for i in range(num_pairs):
        x = [castor[0][i], pollux[0][i]]
        y = [castor[1][i], pollux[1][i]]
        z = [castor[2][i], pollux[2][i]]
        color = next(colors)
        ax.scatter(castor[0][i], castor[1][i], castor[2][i], color=color, s=50, marker='x', zorder=5)
        ax.scatter(pollux[0][i], pollux[1][i], pollux[2][i], color=color, s=50, marker='o', zorder=5)
        ax.plot3D(x, y, z, color=color, linestyle='--', zorder=1, alpha=0.5, label=f"Pair{i}")
    ax.legend(bbox_to_anchor=(1.25, 1))
    plt.title(f"{index}")
    plt.savefig(f"./figures/fig_{str(index).zfill(5)}.png", bbox_inches='tight')
    plt.close()


def make_gif():
    frames = []
    for index in range(5000):
        try:
            frames.append(Image.open(f"./figures/fig_{str(index).zfill(5)}.png"))
        except FileNotFoundError:
            break
    frame_one = frames[0]
    frame_one.save("./figures/gif/result.gif", format="GIF", append_images=frames, save_all=True, duration=100, loop=0)


if __name__ == "__main__":
    visualize()
    make_gif()