import sys
import time
import cupy as cp

from mrcio import parse_map, write_map

if __name__ == '__main__':
    cp.cuda.Device(0).use()

    t0 = time.time()
    data, origin, _, voxel_size = parse_map(sys.argv[1], False, 1.0)
    t1 = time.time()

    print("Time = {:.4f}".format(t1 - t0))

    write_map(sys.argv[2], data, voxel_size, origin=origin)

