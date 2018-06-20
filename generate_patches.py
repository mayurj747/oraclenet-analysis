import numpy as np
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt


def main(cx, cy, filename):
    fig, ax = plt.subplots()
    ax.set_xlim(0,  cx)
    ax.set_ylim(0,  cy)

    polygon = np.asarray(np.load(filename))

    patches = []
    for i in range(polygon.shape[0]):
        obj = Polygon(polygon[i, :, :], True)
        patches.append(obj)

    p = PatchCollection(patches, alpha=1)
    p.set_facecolor(c='#9370DB')
    p.set_edgecolor(c='face')
    ax.add_collection(p)
    ax.set_aspect('equal')


if __name__ == "__main__":


    """
    Converts a set of polygon vertices into pretty patches for clean viz.
    
    """

    load_polys = './random_squares_2.npy'
    main(cx=100, cy=100, filename=load_polys)
