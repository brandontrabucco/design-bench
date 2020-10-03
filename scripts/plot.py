import numpy as np
import argparse
import matplotlib.pyplot as plt


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Plot')
    parser.add_argument('--file', type=str, default='ant_morphology_y.npy')
    parser.add_argument('--bins', type=int, default=1000)
    args = parser.parse_args()

    plt.hist(np.load(args.file), args.bins)
    plt.title(args.file)
    plt.xlabel('Score')
    plt.ylabel('Number Of Examples')
    plt.savefig(args.file + '.png')
    plt.show()
