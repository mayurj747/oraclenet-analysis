import numpy as np
from generate_patches import main as generate_patches
from rrt_star_2D import main as rrt_star
from multiprocessing import Pool
import matplotlib.pyplot as plt
import random
import time


"""
Create a training set using desired planner (here RRT*)
Leveraging multiprocessing to speed up creation.
"""

def viz_pathset(path_set):
    for i in range(len(path_set)):
        path = path_set[i]
        plt.plot(path[:, 0], path[:, 1], 'k')
        plt.plot(path[0, 0], path[0, 1], 'g.', markersize = 10)
        plt.plot(path[-1, 0], path[-1, 1], 'r.', markersize = 10)

def generate_paths(iterator):
    """"
    Interesting note: Numpy's RNG seems to duplicate itself for each process, so each process starts
    from the same defaulkt seed. As a work around, use inbuilt python randint or reset numpy seed for
    each process (using randint, I guess)
    """
    if iterator%100 == 0: print('iterator stage: ', iterator)
    idx0, idx1 = random.randint(0, samples.shape[0]-1), random.randint(0, samples.shape[0]-1)
    path, _ = rrt_star(cx=100, cy=100, start_raw=samples[idx0],
                          goal_raw=samples[idx1], filename=obs_filename,
                          EPSILON=5, plot=False)
    return path


if __name__ == '__main__':
    cx = cy = 100
    plt.close('all')
    obs_filename = 'random_squares_1.npy'
    samples = np.load('random_squares_1_1000samples.npy')
    generate_patches(cx, cy, obs_filename)

    num_paths = 5000
    pool = Pool(10)
    e = []
    tstart = time.time()
    e = pool.map(generate_paths, range(num_paths))
    print('time taken to create dataset', time.time() - tstart)

    path_set = [e[i] for i in range(len(e))]
    path_set = np.asarray(path_set)

    viz_pathset(path_set)  # for an ocular patdown of path_set (▀̿Ĺ̯▀̿ ̿)

    # np.save('training_data_5k_r_sq_1.npy', path_set)
