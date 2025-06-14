import numpy as np
import matplotlib.pyplot as plt

def load_points3D(path):
    points = []
    colors = []
    with open(path, 'r') as f:
        for line in f:
            if line.startswith("#") or line.strip() == "":
                continue
            elems = line.strip().split()
            x, y, z = map(float, elems[1:4])
            r, g, b = map(int, elems[4:7])
            points.append([x, y, z])
            colors.append([r / 255.0, g / 255.0, b / 255.0])  # normalize RGB
    return np.array(points), np.array(colors)

def visualize_points3D(points, colors=None, sample_rate=1):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    sampled_points = points[::sample_rate]
    if colors is not None:
        sampled_colors = colors[::sample_rate]
    else:
        sampled_colors = 'blue'

    ax.scatter(sampled_points[:, 0],
               sampled_points[:, 1],
               sampled_points[:, 2],
               s=0.5, c=sampled_colors, depthshade=True)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=30, azim=-90)
    plt.tight_layout()
    plt.title("3D Points from points3D.txt")
    plt.show()

# Example usage
points3D_path = "../hand_test/0/sparse/0/points3D.txt"
points, colors = load_points3D(points3D_path)
visualize_points3D(points, colors, sample_rate=1)