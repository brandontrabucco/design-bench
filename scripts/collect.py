import numpy as np
import argparse
import glob


if __name__ == '__main__':

    parser = argparse.ArgumentParser('AntData')
    parser.add_argument('--pattern', type=str, default='ant_morphology_X_*.npy')
    parser.add_argument('--out', type=str, default='ant_morphology_X.npy')
    args = parser.parse_args()

    out = []
    for f in sorted(glob.glob(args.pattern)):
        tensor = np.load(f)
        if not any([np.all(np.equal(x.shape, tensor.shape)) and
                    np.all(np.equal(np.abs(x - tensor), 0)) for x in out]):
            out.append(tensor)

    x = np.concatenate(out, axis=0)
    print(x.shape)
    np.save(args.out, x)
