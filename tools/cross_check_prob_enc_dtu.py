import argparse, sys, cv2, os
from this import d
import numpy as np
# from plyfile import PlyData, PlyElement
from PIL import Image
from functools import partial
from multiprocessing import Pool
sys.path.append(".")
from datasets.data_io import read_pfm, save_pfm
from tools.utils import print_args, visualize_depth


parser = argparse.ArgumentParser(description='Dynamic check and filter pseudo depth maps for DTU dataset.')

parser.add_argument('--testpath', help='testing data path')
parser.add_argument('--pairpath', help='pair file path')
parser.add_argument('--pseudo_depth_path', help='path to pseudo depth')
parser.add_argument('--testlist', help='testing scan list')

parser.add_argument('--outdir', help='output dir')
parser.add_argument('--photo_threshold', type=float, default=0.1, help='photo threshold for filter confidence')
parser.add_argument('--geometry_threshold', type=float, default=0.3, help='photo threshold for filter confidence')
parser.add_argument('--thres_view', type=int, default=3, help='threshold of num view')

# parse arguments and check
args = parser.parse_args()
print("argv:", sys.argv[1:])
print_args(args)

# read intrinsics and extrinsics
def read_camera_parameters(filename,scale,index,flag):
    with open(filename) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))

    intrinsics[:2, :] *= scale

    if (flag==0):
        intrinsics[0,2]-=index
    else:
        intrinsics[1,2]-=index

    return intrinsics, extrinsics

# read an image
def read_img(filename):
    img = Image.open(filename)
    # scale 0~255 to 0~1
    np_img = np.array(img, dtype=np.float32) / 255.
    return np_img

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
        for view_idx in range(num_viewpoint):
            ref_view = int(f.readline().rstrip())
            src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
            if len(src_views) == 0:
                continue
            elif len(src_views) < 5:
                continue
            data.append((ref_view, src_views))
    return data


# project the reference point cloud into the source view, then project back
def reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
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

def check_geometric_consistency(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
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
    masks=[]
    for i in range(2,11):
        # mask = np.logical_and(dist < i/4, relative_depth_diff < i/1000)
        mask = np.logical_and(dist < 1, relative_depth_diff < i/1800)
        masks.append(mask)
    vis_mask = np.logical_and(dist < 1, relative_depth_diff < 0.01)
    depth_reprojected[~mask] = 0

    return masks, mask, depth_reprojected, x2d_src, y2d_src, vis_mask


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


def filter_depth(scan_folder, pseudo_folder, out_folder, pair_path, photo_threshold):
    # the pair file
    # pair_file = os.path.join(pair_path, scan_folder, "pair.txt")
    pair_file = pair_path
    # for the final point cloud
    vertexs = []
    vertex_colors = []

    pair_data = read_pair_file(pair_file)

    for ref_view, src_views in pair_data:
        # load the reference image
        ref_img = read_img(os.path.join(scan_folder, 'images/{:0>8}.jpg'.format(ref_view)))
        # load the estimated depth of the reference view
        ref_depth_est = read_pfm(os.path.join(pseudo_folder, 'depth_est/{:0>8}.pfm'.format(ref_view)))[0]
        # ref_depth_est = read_pfm(os.path.join(pseudo_folder, '{:0>8}.pfm'.format(ref_view)))[0]

        # load the photometric mask of the reference view
        confidence = read_pfm(os.path.join(scan_folder, 'confidence/{:0>8}.pfm'.format(ref_view)))[0]

        scale=float(confidence.shape[0])/ref_img.shape[0]
        index=int((int(ref_img.shape[1]*scale)-confidence.shape[1])/2)
        index_p=(int(ref_img.shape[1]*scale)-confidence.shape[1])-index
        flag=0
        if (confidence.shape[1]/ref_img.shape[1]>scale):
            scale=float(confidence.shape[1])/ref_img.shape[1]
            index=int((int(ref_img.shape[0]*scale)-confidence.shape[0])/2)
            index_p=(int(ref_img.shape[0]*scale)-confidence.shape[0])-index
            flag=1

        ref_img=cv2.resize(ref_img,(int(ref_img.shape[1]*scale),int(ref_img.shape[0]*scale)))
        if (flag==0):
            ref_img=ref_img[:,index:ref_img.shape[1]-index_p,:]
        else:
            ref_img=ref_img[index:ref_img.shape[0]-index_p,:,:]

        # load the camera parameters
        ref_intrinsics, ref_extrinsics = read_camera_parameters(
                            os.path.join(scan_folder, 'cams/{:0>8}_cam.txt'.format(ref_view)),scale,index,flag)

        photo_mask = confidence > photo_threshold

        all_srcview_depth_ests = []
        raw_depth_imgs = []

        # compute the geometric mask
        geo_mask_sum = 0
        geo_mask_sums=[]
        vis_masks=[]
        n=1
        for src_view in src_views:
          n+=1
        ct = 0
        for src_view in src_views:
            ct = ct + 1
            src_depth_est = read_pfm(os.path.join(pseudo_folder, 'depth_est/{:0>8}.pfm'.format(src_view)))[0]
            # src_depth_est = read_pfm(os.path.join(pseudo_folder, '{:0>8}.pfm'.format(src_view)))[0]

            src_intrinsics, src_extrinsics = read_camera_parameters(
                os.path.join(scan_folder, 'cams/{:0>8}_cam.txt'.format(src_view)),scale,index,flag)

            masks, geo_mask, depth_reprojected, x2d_src, y2d_src, vis_mask= check_geometric_consistency(ref_depth_est, ref_intrinsics,
                                                                                        ref_extrinsics,
                                                                                        src_depth_est,
                                                                                        src_intrinsics, src_extrinsics)

            vis_masks.append(vis_mask*src_view)

            if (ct==1):
                for i in range(2,n):
                    geo_mask_sums.append(masks[i-2].astype(np.int32))
            else :
                for i in range(2,n):
                    geo_mask_sums[i-2]+=masks[i-2].astype(np.int32)

            geo_mask_sum+=geo_mask.astype(np.int32)

            all_srcview_depth_ests.append(depth_reprojected)
            raw_depth_imgs.append(depth_reprojected)

        # sigma img
        raw_depth_imgs.append(ref_depth_est)
        raw_depth_imgs = np.stack(raw_depth_imgs)   # N,H,W

        geo_mask=geo_mask_sum>=n

        for i in range (2,n):
            geo_mask=np.logical_or(geo_mask,geo_mask_sums[i-2]>=i)
            print(geo_mask.mean())

        depth_est_averaged = (sum(all_srcview_depth_ests) + ref_depth_est) / (geo_mask_sum + 1)

        if (not isinstance(geo_mask, bool)):

            final_mask = np.logical_and(photo_mask, geo_mask)
            depth_o = np.array(depth_est_averaged, dtype=np.float32) * np.array(final_mask, dtype=np.float32)

            final_sigma_img = compute_sigma(raw_depth_imgs, geo_mask_sum+1, depth_o, np.array(final_mask, dtype=np.float32) )

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


def worker(scan):
    scan_folder = os.path.join(args.testpath, scan)
    pseudo_depth_folder = os.path.join(args.pseudo_depth_path, scan)
    out_folder = os.path.join(args.outdir, scan)
    pair_path = os.path.join(args.pairpath, 'Cameras/pair.txt')
    # scan_id = int(scan[4:])
    photo_threshold=args.photo_threshold
    filter_depth(scan_folder, pseudo_depth_folder, out_folder, pair_path, photo_threshold)


if __name__ == '__main__':

    scans = os.listdir(args.testpath)
    testlist = scans
    # testlist = scans[:1]
    partial_func = partial(worker)

    p = Pool()
    p.map(partial_func, testlist)
    p.close()
    p.join()
