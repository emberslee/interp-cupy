import numpy as np
import mrcfile
from interp3d_cupy import Interp3d

def read_mrc(filename, ignorestart=False):
    return read_map(filename, ignorestart)

def read_map(filename, ignorestart=False):
    mrc = mrcfile.open(filename, mode='r')
    data = np.asarray(mrc.data.copy(), dtype=np.float32)
    voxel_size = np.asarray([mrc.voxel_size.x, mrc.voxel_size.y, mrc.voxel_size.z], dtype=np.float32)
    ncrsstart = np.asarray([mrc.header.nxstart, mrc.header.nystart, mrc.header.nzstart], dtype=np.float32)
    origin = np.asarray([mrc.header.origin.x, mrc.header.origin.y, mrc.header.origin.z], dtype=np.float32)
    ncrs = (mrc.header.nx, mrc.header.ny, mrc.header.nz)
    angle = np.asarray([mrc.header.cellb.alpha, mrc.header.cellb.beta, mrc.header.cellb.gamma], dtype=np.float32)
    mapcrs = np.asarray([mrc.header.mapc, mrc.header.mapr, mrc.header.maps], dtype=np.int32)
    mrc.close()

    assert(np.all(angle == 90.0))

    ''' reorder axes

        mapcrs-1    sort        transpose
        0, 1, 2 --> 0, 1, 2 --> 0, 1, 2
        0, 2, 1 --> 0, 2, 1 --> 1, 0, 2
        1, 0, 2 --> 1, 0, 2 --> 0, 2, 1
        1, 2, 0 --> 2, 0, 1 --> 1, 2, 0
        2, 0, 1 --> 1, 2, 0 --> 2, 0, 1
        2, 1, 0 --> 2, 1, 0 --> 2, 1, 0

    '''
    sort = np.asarray([0, 1, 2], dtype=np.int32)
    for i in range(3):
        sort[mapcrs[i] - 1] = i
    nxyzstart = np.asarray([ncrsstart[i] for i in sort], dtype=np.int32)
    nxyz = np.asarray([ncrs[i] for i in sort], dtype=np.int32)
    data = np.transpose(data, axes=2-sort[::-1])

    ''' shift map origins '''
    if not ignorestart:
        origin += np.multiply(nxyzstart, voxel_size)

    return data, origin, voxel_size

def write_mrc(file_name, map, voxel_size, origin=(0.0, 0.0, 0.0), nxyzstart=(0, 0, 0)):
    return write_map(file_name, map, voxel_size, origin, nxyzstart)


def write_map(file_name, map, voxel_size, origin=(0.0, 0.0, 0.0), nxyzstart=(0, 0, 0)):
    mrc = mrcfile.new(file_name, overwrite=True)
    mrc.set_data(map)
    (mrc.header.nxstart, mrc.header.nystart, mrc.header.nzstart) = nxyzstart
    (mrc.header.origin.x, mrc.header.origin.y, mrc.header.origin.z) = origin
    mrc.voxel_size = [voxel_size[i] for i in range(3)]
    mrc.close()


def parse_map(map_file, ignorestart, apix=None, origin_shift=None):

    ''' parse mrc '''
    mrc = mrcfile.open(map_file, mode='r')

    map = np.asfarray(mrc.data.copy(), dtype=np.float32)
    voxel_size = np.asarray([mrc.voxel_size.x, mrc.voxel_size.y, mrc.voxel_size.z], dtype=np.float32)
    ncrsstart = np.asarray([mrc.header.nxstart, mrc.header.nystart, mrc.header.nzstart], dtype=np.float32)
    origin = np.asarray([mrc.header.origin.x, mrc.header.origin.y, mrc.header.origin.z], dtype=np.float32)
    ncrs = (mrc.header.nx, mrc.header.ny, mrc.header.nz)
    angle = np.asarray([mrc.header.cellb.alpha, mrc.header.cellb.beta, mrc.header.cellb.gamma], dtype=np.float32)

    ''' check orthogonal '''
    try:
        assert(angle[0] == angle[1] == angle[2] == 90.0)
    except AssertionError:
        print("# Input grid is not orthogonal. EXIT.")
        mrc.close()
        sys.exit()

    ''' reorder axes '''
    mapcrs = np.subtract([mrc.header.mapc, mrc.header.mapr, mrc.header.maps], 1)
    sort = np.asarray([0, 1, 2], dtype=np.int32)
    for i in range(3):
        sort[mapcrs[i]] = i
    nxyzstart = np.asarray([ncrsstart[i] for i in sort])
    nxyz = np.asarray([ncrs[i] for i in sort])

    map = np.transpose(map, axes=2-sort[::-1])
    mrc.close()

    ''' shift origin according to n*start '''
    if not ignorestart:
        origin += np.multiply(nxyzstart, voxel_size)

    ''' shift by decimal '''
    if origin_shift is not None: 
        origin_shift = origin_shift - origin + np.floor(origin)

    ''' interpolate grid interval '''
    interp3d = Interp3d()
    if apix is not None:
        try:
            assert(voxel_size[0] == voxel_size[1] == voxel_size[2] == apix and origin_shift is None)
        except AssertionError:
            interp3d.del_mapout()
            target_voxel_size = np.asarray([apix, apix, apix], dtype=np.float32)
            print("# Rescale voxel size from {} to {}, shift origin by {}".format(voxel_size, target_voxel_size, origin_shift))
            if origin_shift is not None:
                interp3d.cubic(map, voxel_size[2], voxel_size[1], voxel_size[0], apix, origin_shift[2], origin_shift[1], origin_shift[0], nxyz[2], nxyz[1], nxyz[0])
                origin += origin_shift
            else:
                interp3d.cubic(map, voxel_size[2], voxel_size[1], voxel_size[0], apix, 0.0, 0.0, 0.0, nxyz[2], nxyz[1], nxyz[0])
                
            map = interp3d.mapout
            nxyz = np.asarray([interp3d.pextx, interp3d.pexty, interp3d.pextz], dtype=np.int32)
            voxel_size = target_voxel_size

    assert(np.all(nxyz == np.asarray([map.shape[2], map.shape[1], map.shape[0]], dtype=np.int32)))

    return map, origin, nxyz, voxel_size

