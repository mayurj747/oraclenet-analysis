import numpy as np
import matplotlib.pyplot as plt
import time

import torch
from torch.autograd import Variable

from generate_samples import main as generate_samples, format_obstacles
from rrt_star_2D import check_intersect
from generate_patches import main as generate_patches

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

def path_validity(path, object_c):
    validity = [check_intersect(path[i, :], path[i+1, :], object_c) for i in range(path.shape[0]-1)]
    return all(validity)


def main(cx, cy, obstacle_path, model, start=None, goal=None, num_evals=1, eval_mode=False, plotopt=False):
    """eval_mode is True when function is
        used for testing performance of the network
        When False, function takes in specified start and goal
        and generates a path
    """

    object_c = format_obstacles(load_poly=obstacle_path)
    valid_counter = 0
    path_set = []
    for i in range(num_evals):
        if eval_mode:
            start, goal = generate_samples(cx, cy, load_polys=obstacle_path, num_samples=2)

        start_ext = start[np.newaxis, np.newaxis, ...]
        goal_ext = goal[np.newaxis, np.newaxis, ...]
        s_pred = np.concatenate((start_ext, goal_ext), axis=-1)
        s_pred_var = Variable(torch.from_numpy(s_pred)).type(FloatTensor)
        goal_var = Variable(torch.from_numpy(goal_ext)).type(FloatTensor)
        # goal_var = Variable(torch.from_numpy(goal)).type(FloatTensor)

        s_path = [start]
        num_points = 0
        tstart = time.time()

        while True:
            out1_var, _ = model(s_pred_var, None, force=True)
            out1 = out1_var.data.cpu().numpy()
            s_path.append(out1[0, 0, :])
            s_pred_var = torch.cat((out1_var, goal_var), dim=-1)

            num_points += 1
            if np.linalg.norm(out1 - goal) < 0.2 or num_points > 50:
                break
        tend = time.time()
        # print('time elapsed for generated path: ', tend-tstart)
        s_path = np.asarray(s_path)
        path_set.append(s_path)
        if plotopt:
            plt.plot(s_path[:, 0], s_path[:, 1], 'k')
            plt.plot(start[0], start[1], 'g.', markersize = 10)
            plt.plot(goal[0], goal[1], 'r.', markersize = 10)

        if path_validity(s_path, object_c): valid_counter += 1
    return path_set, valid_counter/num_evals

if __name__ == '__main__':

    from oraclenet_rnn_v1 import SimpleRNN, ProcessData

    data_filename = 'training_data_5k_r_sq_1.npy'
    trainingData = ProcessData(filename=data_filename)
    train_X, train_Y = trainingData.formatData(print_shapes=True)

    model = SimpleRNN(inp_dim=train_X.shape[-1],
                      hidden_size=200,
                      op_dim=train_Y.shape[-1],
                      stacked_layers=1)
    model.cuda()
    model.load_state_dict(torch.load('./Models/test_oraclenet.pkl'))
    print('Loaded model architecture: ', model)

    cx = cy = 100
    obstacle_path = 'random_squares_1.npy'
    generate_patches(cx, cy, obstacle_path)
    _, validity = main(cx, cy, obstacle_path, model, num_evals=100, eval_mode=True, plotopt=True)
    print('Success Rate is ', validity)