import argparse, os, time, sys, gc, cv2, signal
import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement
from multiprocessing import Pool
from functools import partial
sys.path.append(".")
from tools.utils import *
from datasets import find_dataset_def
from datasets.data_io import read_pfm, save_pfm


parser = argparse.ArgumentParser(description='Normal version of depth filtering and probability encoding')
parser.add_argument('--testpath', help='testing data dir')
parser.add_argument('--pairpath', help='pair file path')
parser.add_argument('--testlist', help='testing scene list')
parser.add_argument('--outdir', default='./outputs', help='output dir')
parser.add_argument('--conf', type=float, default=0.05, help='prob confidence')
parser.add_argument('--reproject_dist', type=int, default=1, help='threshold of reprojection dist')
parser.add_argument('--depth_diff', type=float, default=0.005, help='threshold of depth difference')
parser.add_argument('--thres_view', type=int, default=5, help='threshold of num view')
# parse arguments and check
args = parser.parse_args()
print("argv:", sys.argv[1:])

print_args(args)


# read intrinsics and extrinsics
def read_camera_parameters(filename):
    with open(filename) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
    # TODO: assume the feature is 1/4 of the original image size
    # intrinsics[:2, :] /= 4
    return intrinsics, extrinsics


# read an image
def read_img(filename):
    img = Image.open(filename)
    # scale 0~255 to 0~1
    np_img = np.array(img, dtype=np.float32) / 255.
    return np_img


# read a binary mask
def read_mask(filename):
    return read_img(filename) > 0.5


# save a binary mask
def save_mask(filename, mask):
    assert mask.dtype == np.bool
    mask = mask.astype(np.uint8) * 255
    Image.fromarray(mask).save(filename)


# read a pair file, [(ref_view1, [src_view1-1, ...]), (ref_view2, [src_view2-1, ...]), ...]
def read_pair_file(filename):
    data = []
    with open(filename) as f:
        num_viewpoint = int(f.readline())
        # 49 viewpoints
        for view_idx in range(num_viewpoint):
            ref_view = int(f.readline().rstrip())
            src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
            if len(src_views) > 0:
                data.append((ref_view, src_views))
    return data

def write_cam(file, cam):
    f = open(file, "w")
    f.write('extrinsic\n')
    for i in range(0, 4):
        for j in range(0, 4):
            f.write(str(cam[0][i][j]) + ' ')
        f.write('\n')
    f.write('\n')

    f.write('intrinsic\n')
    for i in range(0, 3):
        for j in range(0, 3):
            f.write(str(cam[1][i][j]) + ' ')
        f.write('\n')

    f.write('\n' + str(cam[1][3][0]) + ' ' + str(cam[1][3][1]) + ' ' + str(cam[1][3][2]) + ' ' + str(cam[1][3][3]) + '\n')

    f.close()
    

# project the reference point cloud into the source view, then project back
def reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref, 
                                depth_src, intrinsics_src, extrinsics_src):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    ## step1. project reference pixels to the source view
    # reference view x, y
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    x_ref, y_ref = x_ref.reshape([-1]), y_ref.reshape([-1])
    # reference 3D space
    xyz_ref = np.matmul(np.linalg.inv(intrinsics_ref),
                        np.vstack((x_ref, y_ref, np.ones_like(x_ref))) * depth_ref.reshape([-1]))
    # source 3D space
    xyz_src = np.matmul(np.matmul(extrinsics_src, np.linalg.inv(extrinsics_ref)),
                        np.vstack((xyz_ref, np.ones_like(x_ref))))[:3]
    # source view x, y
    K_xyz_src = np.matmul(intrinsics_src, xyz_src)
    xy_src = K_xyz_src[:2] / K_xyz_src[2:3]

    ## step2. reproject the source view points with source view depth estimation
    # find the depth estimation of the source view
    x_src = xy_src[0].reshape([height, width]).astype(np.float32)
    y_src = xy_src[1].reshape([height, width]).astype(np.float32)
    sampled_depth_src = cv2.remap(depth_src, x_src, y_src, interpolation=cv2.INTER_LINEAR)
    # mask = sampled_depth_src > 0

    # source 3D space
    # NOTE that we should use sampled source-view depth_here to project back
    xyz_src = np.matmul(np.linalg.inv(intrinsics_src),
                        np.vstack((xy_src, np.ones_like(x_ref))) * sampled_depth_src.reshape([-1]))
    # reference 3D space
    xyz_reprojected = np.matmul(np.matmul(extrinsics_ref, np.linalg.inv(extrinsics_src)),
                                np.vstack((xyz_src, np.ones_like(x_ref))))[:3]
    # source view x, y, depth
    depth_reprojected = xyz_reprojected[2].reshape([height, width]).astype(np.float32)
    K_xyz_reprojected = np.matmul(intrinsics_ref, xyz_reprojected)
    xy_reprojected = K_xyz_reprojected[:2] / K_xyz_reprojected[2:3]
    x_reprojected = xy_reprojected[0].reshape([height, width]).astype(np.float32)
    y_reprojected = xy_reprojected[1].reshape([height, width]).astype(np.float32)

    return depth_reprojected, x_reprojected, y_reprojected, x_src, y_src


def check_geometric_consistency(depth_ref, intrinsics_ref, extrinsics_ref, 
                                        depth_src, intrinsics_src, extrinsics_src):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    depth_reprojected, x2d_reprojected, y2d_reprojected, x2d_src, y2d_src = reproject_with_depth(depth_ref, 
                                                                                                 intrinsics_ref, 
                                                                                                 extrinsics_ref, 
                                                                                                 depth_src, 
                                                                                                 intrinsics_src, 
                                                                                                 extrinsics_src)
    # check |p_reproj-p_1| < 1
    dist = np.sqrt((x2d_reprojected - x_ref) ** 2 + (y2d_reprojected - y_ref) ** 2)

    # check |d_reproj-d_1| / d_1 < 0.01
    depth_diff = np.abs(depth_reprojected - depth_ref)
    relative_depth_diff = depth_diff / depth_ref

    # mask = np.logical_and(dist < 1, relative_depth_diff < 0.01)
    mask = np.logical_and(dist < args.reproject_dist, relative_depth_diff < args.depth_diff)
    depth_reprojected[~mask] = 0

    return mask, depth_reprojected, x2d_src, y2d_src


def compute_sigma(raw_depth_imgs, geo_mask_sums, depth_average, final_mask):
    '''Compute the sigma image from the stack imgs
    : param: raw_depth_imgs, N,H,W
    : param: feo_mask_sums, H,W
    '''
    depth_average = depth_average *  final_mask
    raw_sigma_mask = np.array((raw_depth_imgs > 0.), dtype=np.float32)  # N,H.W
    final_sigma_img = (raw_depth_imgs[:,] - depth_average)**2 * raw_sigma_mask  *  final_mask  # N,H,W
    final_sigma_img = np.sum(final_sigma_img, axis=0) / geo_mask_sums     # H,W
    final_sigma_img = np.array(np.sqrt(final_sigma_img), dtype=np.float32) * final_mask# * 0.1
    return final_sigma_img  # H,W



def filter_depth(pair_folder, scan_folder, out_folder):
    # the pair file
    # pair_file = os.path.join(pair_folder, "pair.txt")
    pair_file = pair_folder
    # for the final point cloud


    pair_data = read_pair_file(pair_file)
    nviews = len(pair_data)

    # for each reference view and the corresponding source views
    for ref_view, src_views in pair_data:
        ref_intrinsics, ref_extrinsics = read_camera_parameters(
            os.path.join(scan_folder, 'cams/{:0>8}_cam.txt'.format(ref_view)))
        # load the reference image
        ref_img = read_img(os.path.join(scan_folder, 'images/{:0>8}.jpg'.format(ref_view)))
        # load the estimated depth of the reference view
        ref_depth_est = read_pfm(os.path.join(scan_folder, 'depth_est/{:0>8}.pfm'.format(ref_view)))[0]
        # load the photometric mask of the reference view
        confidence = read_pfm(os.path.join(scan_folder, 'confidence/{:0>8}.pfm'.format(ref_view)))[0]
        photo_mask = confidence > args.conf

        all_srcview_depth_ests = []
        raw_depth_imgs = []
        all_srcview_x = []
        all_srcview_y = []
        all_srcview_geomask = []

        # compute the geometric mask
        geo_mask_sum = 0
        
        for src_view in src_views:
            # camera parameters of the source view
            src_intrinsics, src_extrinsics = read_camera_parameters(
                os.path.join(scan_folder, 'cams/{:0>8}_cam.txt'.format(src_view)))
            # the estimated depth of the source view
            src_depth_est = read_pfm(os.path.join(scan_folder, 'depth_est/{:0>8}.pfm'.format(src_view)))[0]

            geo_mask, depth_reprojected, x2d_src, y2d_src = check_geometric_consistency(ref_depth_est, ref_intrinsics, 
                                                                        ref_extrinsics,
                                                                      src_depth_est,
                                                                      src_intrinsics, src_extrinsics)
            geo_mask_sum += geo_mask.astype(np.int32)
            all_srcview_depth_ests.append(depth_reprojected)
            raw_depth_imgs.append(depth_reprojected)
            all_srcview_x.append(x2d_src)
            all_srcview_y.append(y2d_src)
            all_srcview_geomask.append(geo_mask)

        depth_est_averaged = (sum(all_srcview_depth_ests) + ref_depth_est) / (geo_mask_sum + 1)
        # at least 3 source views matched
        geo_mask = geo_mask_sum >= args.thres_view
        final_mask = np.logical_and(photo_mask, geo_mask)
        
        raw_depth_imgs.append(ref_depth_est)
        raw_depth_imgs = np.stack(raw_depth_imgs)
        
        depth_o = np.array(depth_est_averaged, 
                                            dtype=np.float32) * np.array(final_mask, dtype=np.float32)
        
        final_sigma_img = compute_sigma(raw_depth_imgs, geo_mask_sum+1, 
                                                                                                depth_o, np.array(final_mask, dtype=np.float32) )
        masked_final_sigma = np.array(final_sigma_img, dtype=np.float32) * np.array(final_mask, dtype=np.float32)
        
        os.makedirs(os.path.join(out_folder, "mask"), exist_ok=True)
        os.makedirs(os.path.join(out_folder, "checked_depth"), exist_ok=True)
        os.makedirs(os.path.join(out_folder, "fused_sigma"), exist_ok=True)
        save_pfm(os.path.join(out_folder, "checked_depth/{:0>8}.pfm".format(ref_view)), depth_o)
        save_pfm(os.path.join(out_folder, "fused_sigma/{:0>8}.pfm".format(ref_view)), masked_final_sigma)
        cv2.imwrite(os.path.join(out_folder, "checked_depth/{:0>8}.png".format(ref_view)), (depth_o/905)*255)
        cv2.imwrite(os.path.join(out_folder, "fused_sigma/{:0>8}.png".format(ref_view)), (masked_final_sigma/masked_final_sigma.max())*255)
        save_mask(os.path.join(out_folder, "mask/{:0>8}_photo.png".format(ref_view)), photo_mask)
        save_mask(os.path.join(out_folder, "mask/{:0>8}_geo.png".format(ref_view)), geo_mask)
        save_mask(os.path.join(out_folder, "mask/{:0>8}_final.png".format(ref_view)), final_mask)

        print("processing {}, ref-view{:0>2}, photo/geo/final-mask:{}/{}/{}".format(scan_folder, ref_view,
                                                                                        photo_mask.mean(),
                                                                                        geo_mask.mean(),
                                                                                        final_mask.mean()))



def init_worker():
    '''
    Catch Ctrl+C signal to termiante workers
    '''
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def pcd_filter_worker(scan):
    # pair_folder = os.path.join(args.testpath, scan)
    pair_path = os.path.join(args.pairpath, 'Cameras/pair.txt')
    scan_folder = os.path.join(args.testpath, scan)
    out_folder = os.path.join(args.outdir, scan)
    filter_depth(pair_path, scan_folder, out_folder)


def pcd_filter(testlist, number_worker):

    partial_func = partial(pcd_filter_worker)

    p = Pool(number_worker, init_worker)
    try:
        p.map(partial_func, testlist)
    except KeyboardInterrupt:
        print("....\nCaught KeyboardInterrupt, terminating workers")
        p.terminate()
    else:
        p.close()
    p.join()

if __name__ == '__main__':

    with open(args.testlist) as f:
            content = f.readlines()
            testlist = [line.rstrip() for line in content]

    pcd_filter(testlist, 4)
    # scan = testlist[0]
    # pair_path = os.path.join(args.pairpath, 'Cameras/pair.txt')
    # scan_folder = os.path.join(args.testpath, scan)
    # out_folder = os.path.join(args.outdir, scan)
    # filter_depth(pair_path, scan_folder, out_folder)
    
