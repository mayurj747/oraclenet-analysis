import numpy as np
from generate_patches import main as generate_patches
import matplotlib.pyplot as plt
import matplotlib.path as mplPath


"""
Create a sample set of collision-free points. Can then use permutations of these points to get pairs
of start and goal points for training set.
"""

def check_collision(points, object_c):
    decisions = [p.contains_point(points) for p in object_c]
    return not any(decisions) # will return True if all points are collision free


if __name__ == '__main__':
    load_polys = 'random_squares_1.npy'
    cx = cy = 100
    plt.close('all')
    generate_patches(cx, cy, load_polys)

    polygon = np.asarray(np.load(load_polys))
    object_c = []
    for i in range(polygon.shape[0]):
        obj = mplPath.Path(polygon[i, :, :])
        object_c.append(obj)

    num_samples = 1000
    sample_set = []
    for i in range(num_samples):
        while True:
            samples = np.random.uniform(0, cx, 2)
            if check_collision(samples, object_c):
                plt.plot(samples[0], samples[1], 'r.')
                sample_set.append(samples)
                break
    np.save('random_squares_1_1000samples.npy', sample_set)    num_samples = 1000
