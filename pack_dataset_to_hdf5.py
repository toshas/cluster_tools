#!/usr/bin/env python

import os
import glob
import h5py
import argparse
import numpy as np
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=str)
    parser.add_argument('name', type=str)
    args, unknown = parser.parse_known_args()
    if unknown is not None and len(unknown) > 0:
        print('Unknown arguments: {}'.format(unknown))

    folder = os.path.expanduser(os.path.expandvars(args.folder))
    assert os.path.isdir(folder)
    list_filenames = glob.glob(folder + '/**/*', recursive=True)
    list_filenames = [f for f in list_filenames if os.path.isfile(f)]

    hdf5_path = os.path.join(folder, args.name)
    assert not os.path.exists(hdf5_path)
    hdf5_file = h5py.File(os.path.join(folder, args.name + '.hdf5'), mode='w')

    t_vlen_uint8 = h5py.special_dtype(vlen=np.uint8)

    for full_path in tqdm(list_filenames):
        key = full_path[len(folder):].strip('/')
        g = hdf5_file.create_group(key)
        ds = g.create_dataset('raw', shape=(1,), dtype=t_vlen_uint8)
        with open(full_path, 'rb') as fp:
            file_content = fp.read()
            file_content = np.frombuffer(file_content, dtype='uint8')
            ds[0] = file_content

    hdf5_file.close()


if __name__ == '__main__':
    main()
