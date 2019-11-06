import argparse
from collections import defaultdict
import h5py
import numpy as np
import tifffile
import skimage


parser = argparse.ArgumentParser('script to generate training data')
parser.add_argument('--image', type=str, default='./raw3_250_top_filter1.0.tif', help='image data path')
parser.add_argument('--label', type=str, default='./label_raw3_final_top.tif', help='label data path')
parser.add_argument('--save', type=str, default='data_raw3_focus_250_filter1_top_area64.h5', help='save file name')
parser.add_argument('--shape', type=list, default=[75, 75, 75], help='seed shape')
parser.add_argument('--thr', type=list, default=[0.025, 0.05, 0.075, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
parser.add_argument('--min_size', type=int, default=10000)

args = parser.parse_args()


def _query_summed_volume(svt, diam):
    """Queries a summed volume table.

    Operates in 'VALID' mode, i.e. only computes the sums for voxels where the
    full diam // 2 context is available.

    Args:
    svt: summed volume table (see _summed_volume_table)
    diam: diameter (z, y, x tuple) of the area within which to compute sums

    Returns:
    sum of all values within a diam // 2 radius (under L1 metric) of every voxel
    in the array from which 'svt' was built.
    """
    return (
        svt[diam[0]:, diam[1]:, diam[2]:] - svt[diam[0]:, diam[1]:, :-diam[2]] -
        svt[diam[0]:, :-diam[1], diam[2]:] - svt[:-diam[0], diam[1]:, diam[2]:] +
        svt[:-diam[0], :-diam[1], diam[2]:] + svt[:-diam[0], diam[1]:, :-diam[2]]
        + svt[diam[0]:, :-diam[1], :-diam[2]] -
        svt[:-diam[0], :-diam[1], :-diam[2]])


def _summed_volume_table(val):
    """Computes a summed volume table of 'val'."""
    val = val.astype(np.int32)
    svt = val.cumsum(axis=0).cumsum(axis=1).cumsum(axis=2)
    return np.pad(svt, [[1, 0], [1, 0], [1, 0]], mode='constant')


def clear_dust(data, min_size=10):
    """Removes small objects from a segmentation array.

    Replaces objects smaller than `min_size` with 0 (background).

    Args:
    data: numpy array of segment IDs
    min_size: minimum size in voxels of an object to be retained

    Returns:
    the data array (modified in place)
    """
    ids, sizes = np.unique(data, return_counts=True)
    small = ids[sizes < min_size]
    small_mask = np.in1d(data.flat, small).reshape(data.shape)
    data[small_mask] = 0
    return data


def compute_partitions(seg_array, thresholds, lom_radius, min_size=10000):
    seg_array = clear_dust(seg_array, min_size=min_size)
    assert seg_array.ndim == 3

    lom_radius = np.array(lom_radius)
    lom_radius_zyx = lom_radius[::-1]
    lom_diam_zyx = 2 * lom_radius_zyx + 1

    def _sel(i):
        if i == 0:
            return slice(None)
        else:
            return slice(i, -i)

    valid_sel = [_sel(x) for x in lom_radius_zyx]
    output = np.zeros(seg_array[valid_sel].shape, dtype=np.uint8)
    corner = lom_radius

    labels = set(np.unique(seg_array))

    fov_volume = np.prod(lom_diam_zyx)
    for l in labels:
        # Don't create a mask for the background component.
        if l == 0:
            continue

        object_mask = (seg_array == l)

        svt = _summed_volume_table(object_mask)
        active_fraction = _query_summed_volume(svt, lom_diam_zyx) / fov_volume
        assert active_fraction.shape == output.shape

        # Drop context that is only necessary for computing the active fraction
        # (i.e. one LOM radius in every direction).s
        object_mask = object_mask[valid_sel]

        # TODO(mjanusz): Use np.digitize here.
        for i, th in enumerate(thresholds):
            output[object_mask & (active_fraction < th) & (output == 0)] = i + 1

        output[object_mask & (active_fraction >= thresholds[-1]) & (output == 0)] = len(thresholds) + 1

    return corner, output


def swc_get_coord(swc_data, id, image_shape,rad,bedx,coor, label):

    train_shape = args.shape[0]
    if (train_shape % 2) ==1:
        train_shape +=1
    margin = train_shape/2


    index = int(id-1)
    x = int(swc_data[index][2])
    y = int(swc_data[index][3])
    z = int(swc_data[index][4])
    for xd in range(-rad, rad+1):
        for yd in range(-rad, rad+1):
            for zd in range(-rad, rad+1):
                if ((z+zd) >= (image_shape[0]-margin)) |((y+yd) >= (image_shape[1]-margin))|((x+xd) >= (image_shape[2]-margin))|((z+zd) <= (0+margin))|((y+yd) <= (0+margin))|((x+xd) <= (0+margin)):
                    continue

                if label[z+zd, y+yd, x+xd] > 0:
                #if ((z+zd) >= 499) |((y+yd) >= (249))|((x+xd) >= (249)):
                    #continue
                    bedx[z + zd, y + yd, x + xd] = 200
                    coor.append(np.array([z+zd, y+yd, x+xd]))





def run():
    images = tifffile.TiffFile(args.image).asarray()
    labels = tifffile.TiffFile(args.label).asarray()
    # images = np.array([cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in images])
    # labels = np.array([cv2.cvtColor(label, cv2.COLOR_BGR2GRAY) for label in labels])



    image_shape = images.shape
    print(image_shape)

    """
    swc_data = np.loadtxt('raw3_swc.swc')
    expansion = 1

    swc_data[:, 2] = swc_data[:, 2] * expansion
    swc_data[:, 3] = swc_data[:, 3] * expansion
    swc_data[:, 4] = swc_data[:, 4] * expansion
    swc_data[:, 4]= swc_data[:, 4] - 49
    swc_data[:, 5] = swc_data[:, 5] * expansion

    swc_data = np.round(swc_data)
    dataNum = len(swc_data[:, 5])
    print("point_num:",dataNum)

    coor = []

    bedx = np.zeros((250, 250, 250))

    for point in range(dataNum):
        id = swc_data[point][0]
        parent = swc_data[point][6]
        topo = id - parent

        if id-parent != 1:
            
            #parent_int = int(parent)
            #rad_p = swc_data[parent_int][5]
          
            #rad = int(rad_p)
           
            #swc_get_coord(swc_data, parent, image_shape, rad, bedx, coor, labels)
            

            for id_all in range(-3,4):
                if (id_all == -1)|(id_all == 1)|(id_all == 0):
                    continue
                swc_get_coord(swc_data,id+id_all, image_shape, 2, bedx, coor,labels)
                swc_get_coord(swc_data, parent+id_all, image_shape, 2, bedx, coor, labels)
    """



    """
    m = np.array([int(x / 2) for x in args.shape])
    seg = labels.copy()
    corner, partitions = compute_partitions(seg[...], [float(x) for x in args.thr], m, args.min_size)

    totals = defaultdict(int)  # partition -> voxel count
    indices = defaultdict(list)  # partition -> [(vol_id, 1d index)]

    # end = np.array(labels.shape) - m
    # sel = [slice(s, e) for s, e in zip(m, end)]
    # partitions = labels[tuple(sel)].copy()
    vol_shapes = partitions.shape
    uniques, counts = np.unique(partitions, return_counts=True)
    for val, cnt in zip(uniques, counts):
        if val == 255 or val == 0:
            continue

        totals[val] += cnt
        indices[val].extend([(0, flat_index) for flat_index in np.flatnonzero(partitions == val)])

    max_count = max(totals.values())
    indices = np.concatenate([np.resize(np.random.permutation(v), (max_count, 2)) for v in indices.values()], axis=0)
    np.random.shuffle(indices)


    for i, coord_idx in indices:
        if (coord_idx % 100) == 99:
            print(coord_idx)
            z, y, x = np.unravel_index(coord_idx, vol_shapes)
            coor.append([z + m[2], y + m[1], x + m[0]])
            bedx[z + m[2], y + m[1], x + m[0]] = 200
    """
    coor = []
    bedx = np.zeros((250, 250, 250))
    coords_m = np.array([[54,153,108],[43,186,135],[48,199,136],[49,200,137],[47,200,151],[34,204,145],[66,153,110],[76,193,48],[93,130,63],[99,90,55],[89,122,71],[100,208,54],[99,186,55],[88,197,46],[88,196,59],[98,194,79],[100,175,128],[103,173,143],[104,168,115],[106,160,124],[102,196,95],[97,212,85],[145,184,76],[140,184,58],[158,184,112],[158,189,126],[160,179,84],[160,193,68],[166,187,105],[166,170,114],[171,158,124],[171,161,120],[171,153,142],[171,152,157],[172,142,122],[174,118,106],[174,134,114],[169,165,140],[169,174,149],[169,183,157],[172,209,186],[171,201,198],[162,197,143],[165,193,131],[158,186,116],[170,177,174],[155,194,197],[155,203,178],[168,204,29],[170,206,29],[171,146,174],[158,209,44],[159,85,46],[159,86,68],[158,95,87],[147,101,67],[180,98,100],[180,113,106],[188,80,90],[199,72,88],[179,73,77],[185,71,98],[188,168,107],[183,94,81],[182,90,87],[180,84,106],[180,80,116],[178,70,134],[158,131,173],[158,148,202],[183,106,162],[178,65,151],[176,136,198],[195,86,163],[204,162,137],[194,142,138],[210,148,152],[69,149,105],[60,67,113],[52,190,73],[97,193,82],[98,172,140],[162,101,101],[162,115,107]])

    train_shape = args.shape[0]
    if (train_shape % 2) == 1:
        train_shape += 1
    margin = train_shape / 2

    cn = len(coords_m)
    for idx in range(cn):
        z = int(coords_m[idx][0])
        y = int(coords_m[idx][1])
        x = int(coords_m[idx][2])
        print("coord", z,y,x)
        rad = 1

        for xd in range(-rad, rad + 1):
            for yd in range(-rad, rad + 1):
                for zd in range(-rad, rad + 1):
                    if ((z + zd) >= (image_shape[0] - margin)) | ((y + yd) >= (image_shape[1] - margin)) | (
                            (x + xd) >= (image_shape[2] - margin)) | ((z + zd) <= (0 + margin)) | (
                            (y + yd) <= (0 + margin)) | ((x + xd) <= (0 + margin)):
                     continue

                    if labels[z + zd, y + yd, x + xd] > 0:
                        # if ((z+zd) >= 499) |((y+yd) >= (249))|((x+xd) >= (249)):
                        # continue
                        bedx[z + zd, y + yd, x + xd] = 200
                        coor.append(np.array([z + zd, y + yd, x + xd]))
                        print("coord_surround", z + zd, y + yd, x + xd)
    print(len(coor))

    skimage.io.imsave('coord_bed10.tif', bedx.astype('uint8'))
    with h5py.File(args.save, 'w') as f:
        f.create_dataset('image', data=images, compression='gzip')
        f.create_dataset('label', data=labels, compression='gzip')
        f.create_dataset('coor', data=coor, compression='gzip')


if __name__ == '__main__':
    run()
