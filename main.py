import sys
import cupy as cp

from mrcio import parse_map, write_map

if __name__ == '__main__':
    cp.cuda.Device(0).use()

    data, origin, _, voxel_size = parse_map(sys.argv[1], False, 1.0)

    write_map(sys.argv[2], data, voxel_size, origin=origin)

