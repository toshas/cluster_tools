# Cluster tools

dotfiles for cluster environments

# Why

Many distributed environments for machine learning impose constraints on the data processing workflow. 

# Overcoming file (inode) limit

One such constraint is the total number of files allowed on distributed storage media. This becomes a problem when working with modern datasets, consisting of millions of little files. 

## Workflow

1. Run `pack_dataset_to_hdf5.py <path-to-dataset> <output-file-name>` to convert a directory with dataset files into a single dataset blob;
2. Modify dataset accessor (e.g., `torch.utils.data.Dataset` instance) by changing `open` calls to working with the blob dataset instead. The following snippet reads out an image file from an HDF5:

```python
import h5py, io
from PIL import Image

def read_binary(rel_path, hdf5_path):
    hfile = None
    try:
        hfile = h5py.File(hdf5_path, 'r')
        return hfile[rel_path]['raw'][0]
    finally:
        if hfile is not None:
            hfile.close()

def read_image(rel_path, hdf5_path):
    bytes = access_item(rel_path, hdf5_path)
    return Image.open(io.BytesIO(bytes)).convert('RGB')
```

## Alternatives
Check the following items before using snippets from this repository:
- tfrecord
- Petastorm
- tar
