import numpy as np
import sys
import time
import h5py
import math
from superkmeans import SuperKMeans


def print_usage(program_name):
    print(f"Usage: {program_name} <hdf5_path> [k] [quantizer]")
    print("  hdf5_path: Path to HDF5 file containing 'train' dataset")
    print("  k: Number of clusters (default: 1000)")
    print("  quantizer: f32, sq8, lvq4, or rabitq (default: f32)")
    print("Example:")
    print(f"  {program_name} data/dataset.hdf5 4000 sq8")


def read_hdf5_data(hdf5_path):
    """Read train and test data from HDF5 file."""
    with h5py.File(hdf5_path, "r") as hdf5_file:
        train = np.array(hdf5_file["train"], dtype=np.float32)
        test = np.array(hdf5_file["test"], dtype=np.float32) if "test" in hdf5_file else None
    return train, test


def main():
    if len(sys.argv) < 2:
        print("Error: Missing HDF5 file path")
        print_usage(sys.argv[0])
        return 1
    if sys.argv[1] in ['-h', '--help']:
        print_usage(sys.argv[0])
        return 0

    hdf5_path = sys.argv[1]

    print(f"Loading data from {hdf5_path}...")
    try:
        data, test = read_hdf5_data(hdf5_path)
    except Exception as e:
        print(f"Error reading HDF5 file: {e}")
        return 1

    n, d = data.shape
    k = int(sys.argv[2]) if len(sys.argv) > 2 else int(math.sqrt(n) * 4)
    quantizer = sys.argv[3] if len(sys.argv) > 3 else "f32"
    print(f"Loaded {n:,} vectors with {d} dimensions")
    print(f"Number of clusters: {k}")
    print(f"Quantizer: {quantizer}")

    kmeans = SuperKMeans(
        n_clusters=k,
        dimensionality=d,
        quantizer=quantizer,
    )

    print("Generating centroids...")
    start_time = time.time_ns()
    centroids = kmeans.train(data)
    end_time = time.time_ns()
    print(f"Index built in {(end_time - start_time) / 1000000:.2f} milliseconds")

    print("Assigning points to clusters...")
    assignments = kmeans.assign(data, centroids)

    print("Done!")



if __name__ == "__main__":
    main()
