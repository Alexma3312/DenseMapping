import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle
import numpy as np

from superpixel_extraction import SuperpixelExtraction
from surfel_generation import SurfelGeneration
from utilities.data_helper import read_ground_truth_poses
from utilities.patch2d_to_3d import pathpatch_2d_to_3d, pathpatch_translate
from surfel_element import SurfelElement
from superpixel_seed import SuperpixelSeed
from typing import List
from save_mesh import generate_pointcloud

def rgb2gray(rgb):
    if (rgb.ndim==3):
        return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
    else:
        return rgb
def gray2rgb(gray):
    return np.stack((gray,gray,gray), axis=2)

def plot_sp(image, superpixel_idx, superpixels, toPlot=True):
    sp_idx_full = cv2.resize(superpixel_idx.astype(np.float), image.shape[1::-1],
                                interpolation=cv2.INTER_NEAREST)
    sp_edges = np.zeros(sp_idx_full.shape, dtype=bool)
    sp_edges[:-1, :] = (np.diff(sp_idx_full, axis=0) != 0)
    sp_edges[:, :-1] = (np.diff(sp_idx_full, axis=1) != 0) | sp_edges[:, :-1]
    image_sp = np.copy(image)
    xs = np.array([sp.x*image.shape[1]/superpixel_idx.shape[1] for sp in superpixels], dtype=int)
    ys = np.array([sp.y*image.shape[0]/superpixel_idx.shape[0] for sp in superpixels], dtype=int)
    if (len(image.shape)==3): # rgb
        image_sp[sp_edges,0] = 1
        image_sp[sp_edges,1] = 0
        image_sp[sp_edges,2] = 0
        image_sp[ys,xs,1] = 1
    else: # depth/grayscale
        image_sp[sp_edges] = np.min(image_sp)
        image_sp[ys,xs] = np.max(image_sp)
    if toPlot:
        plt.imshow(image_sp)
        # plt.scatter(
        #     [sp.x*image.shape[0]/superpixel_idx.shape[0] for sp in superpixels],
        #     [sp.y*image.shape[0]/superpixel_idx.shape[0] for sp in superpixels], s=1)
        plt.draw();plt.pause(0.001)
    return image_sp

def plot_superpixels(all_superpixels: List[SuperpixelSeed]):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    all_x = [sp.posi_x for sp in all_superpixels]
    all_y = [sp.posi_y for sp in all_superpixels]
    all_z = [sp.posi_z for sp in all_superpixels]
    ax.scatter(all_x, all_y, all_z)

    plt.show()

def plot_surfels(all_surfels: List[SurfelElement]):
    print('plotting surfels...')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    all_sizes = np.array([surfel.size for surfel in all_surfels])
    all_sizes[np.isinf(all_sizes)] = np.nan

    all_surfels_trans = []
    for surfel in all_surfels:
        surfel_t = surfel
        surfel_t.nx, surfel_t.ny, surfel_t.nz = surfel_t.nx, surfel_t.nz, -surfel_t.ny
        surfel_t.px, surfel_t.py, surfel_t.pz = surfel_t.px, surfel_t.pz, -surfel_t.py
        all_surfels_trans.append(surfel_t)
    for surfel in all_surfels_trans:
        if surfel.update_times < 3:
            continue
        # p = Circle((0, 0), 0*0.01 + 1*surfel.size / np.nanmax(all_sizes), facecolor=(surfel.color,)*3, alpha=.8)
        p = Circle((0, 0), surfel.size, facecolor=(surfel.color,)*3, alpha=.8)
        ax.add_patch(p)
        pathpatch_2d_to_3d(p, z=0, normal=(surfel.nx, surfel.ny, surfel.nz))
        pathpatch_translate(p, (surfel.px, surfel.py, surfel.pz))
    
    # calculate bounds
    mins = [1e9, 1e9, 1e9]
    maxs = [-1e9, -1e9, -1e9]
    for surfel in all_surfels_trans:
        mins[0] = min(mins[0], surfel.px)
        mins[1] = min(mins[1], surfel.py)
        mins[2] = min(mins[2], surfel.pz)
        maxs[0] = max(maxs[0], surfel.px)
        maxs[1] = max(maxs[1], surfel.py)
        maxs[2] = max(maxs[2], surfel.pz)
    rs = [maxs[i]-mins[i] for i in range(3)]
    mins = [m-0.1*r for m,r in zip(mins, rs)]
    maxs = [m+0.1*r for m,r in zip(maxs, rs)]
    ax.set_xlim([mins[0], maxs[0]])
    ax.set_ylim([mins[1], maxs[1]])
    ax.set_zlim([mins[2], maxs[2]])

    print('done plotting')
    plt.show()

def get_all_superpixels(folder_rgb, folder_depth, filenames, folder_cache='../dataset/superpixels/'):
    print('Extracting superpixels for all frames...')
    all_superpixels = []
    for i, filename in enumerate(filenames):
        try:
            superpixels = np.load(folder_cache+filename[:-4]+'.npy')
            print('loaded frame {:d} of {:d} ('.format(i+1, len(filenames)) + filename + ') from cache')
            all_superpixels.append(superpixels)
            continue
        except FileNotFoundError:
            pass
        print('now processing frame {:d} of {:d} ('.format(i+1, len(filenames)) + filename + ')')
        image_full = plt.imread(folder_rgb + filename)
        if (image_full.ndim < 3):
            image_full = gray2rgb(image_full)
        depth_full = plt.imread(folder_depth + filename)# / 5000
        depth_full = rgb2gray(depth_full) * 2**16 / 5000
        scale = 0.5
        image = cv2.resize(rgb2gray(image_full),
            (int(np.size(image_full,1)*scale), int(np.size(image_full,0)*scale)))
        depth = cv2.resize(depth_full,
            (int(np.size(depth_full,1)*scale), int(np.size(depth_full,0)*scale)))# * scale
        imgray = rgb2gray(image)
        camera_parameters = {'fx': 525*scale, 'fy': 525*scale,
                            'cx': 319.5*scale, 'cy': 239.5*scale}
        weights = {'Ns': 200, 'Nc': 2, 'Nd': 5}
        spExtractor = SuperpixelExtraction(imgray, depth, camera_parameters,
                                        weights=weights, sp_size=int(12*scale))

        superpixels = spExtractor.extract_superpixels(iterations=5)
        all_superpixels.append(superpixels)
        if folder_cache is not None:
            np.save(folder_cache+filename[:-4], superpixels)
    return all_superpixels

def get_all_surfels(all_superpixels, indexes=None):
    if (indexes == None):
        indexes = [i for i in range(len(filenames))]
    print('Fusing all surfels...')
    poses = read_ground_truth_poses()
    scale = 0.5
    camera_parameters = {'fx': 525*scale, 'fy': 525*scale,
                        'cx': 319.5*scale, 'cy': 239.5*scale}
    sg = SurfelGeneration(camera_parameters, MAX_ANGLE_COS=0.25)
    for i, superpixels in enumerate(all_superpixels):
        print('updating surfels for frame #{:d} ({:d} of {:d})'.format(indexes[i], i+1, len(all_superpixels)))
        sg.update_surfels(i, list(superpixels), poses[indexes[i]])
        # sg.update_surfels(i, list(superpixels), poses[0])
        # print(poses[indexes[i]])
    return sg

def main():
    folder_rgb = '../dataset/rgb/'
    folder_depth = '../dataset/depth/'
    # indexes = [i for i in range(0, 400, 100)]
    # indexes = [i for i in range(0, 15, 3)]
    indexes = [i for i in range(0, 250, 10)]
    filenames = ['{:d}.png'.format(i) for i in indexes]

    if False:
        all_superpixels = np.load('../dataset/results/all_superpixels_good.npy')
    else:
        all_superpixels = get_all_superpixels(folder_rgb, folder_depth, filenames,
                                              folder_cache='../dataset/superpixels_fine/')
        # np.save('../dataset/results/all_superpixels_good', all_superpixels)
    if False:
        all_surfels = np.load('../dataset/results/surfels.npy').all_surfels
    else:
        sg = get_all_surfels(all_superpixels, indexes=indexes)
        np.save('../dataset/results/surfels', [sg])
        all_surfels = sg.all_surfels
    # plot_superpixels(all_superpixels[0])
    generate_pointcloud(all_surfels, '../dataset/test.ply')
    # plot_surfels(all_surfels)

def main_old():
    image_full = plt.imread('../dataset/rgb/0.png')
    # image_full = gray2rgb(image_full)
    depth_full = plt.imread('../dataset/depth/0.png')# / 5000
    depth_full = rgb2gray(depth_full) * 2**16 / 5000
    scale = 0.5
    image = cv2.resize(image_full,
        (int(np.size(image_full,1)*scale), int(np.size(image_full,0)*scale)))
    depth = cv2.resize(depth_full,
        (int(np.size(depth_full,1)*scale), int(np.size(depth_full,0)*scale)))# * scale
    imgray = rgb2gray(image)
    camera_parameters = {'fx': 525*scale, 'fy': 525*scale,
                         'cx': 319.5*scale, 'cy': 239.5*scale}
    weights = {'Ns': 200, 'Nc': 2, 'Nd': 5}
    spExtractor = SuperpixelExtraction(imgray, depth, camera_parameters,
                                       weights=weights, sp_size=int(50*scale))
    
    # original images
    plt.figure(1, figsize=(18,8))
    plt.subplot(2,1,1)
    plt.imshow(image_full)
    plt.subplot(2,1,2)
    plt.imshow(depth_full)
    plt.draw();plt.pause(0.001)
    # superpixels
    superpixels = spExtractor.init_seeds()
    for i in range(5):
        superpixel_idx = spExtractor.assign_pixels(superpixels)
        superpixels = spExtractor.update_seeds(superpixel_idx, superpixels)
        # plt.subplot(2,1,1)
        image_sp = plot_sp(image_full, superpixel_idx, superpixels, toPlot=False)
        # plt.subplot(2,1,2)
        depth_sp = plot_sp(depth_full, superpixel_idx, superpixels, toPlot=False)
        # plt.imsave('../dataset/results/kitti_superpixels_rgb{:02d}'.format(i), image_sp)
        # plt.imsave('../dataset/results/kitti_superpixels_depth{:02d}'.format(i), depth_sp)
    superpixels = spExtractor.calc_norms(superpixel_idx, superpixels)

    # # plotting
    plt.subplot(2,1,1)
    image_sp = plot_sp(image_full, superpixel_idx, superpixels)
    plt.subplot(2,1,2)
    depth_sp = plot_sp(depth_full, superpixel_idx, superpixels)

    # plt.imsave('../dataset/results/kitti_superpixels_rgb', image_sp)
    # plt.imsave('../dataset/results/kitti_superpixels_depth', depth_sp)

    plt.figure(2)
    ax1 = plt.subplot(1,1,1, projection='3d')
    xs = np.array([sp.posi_x for sp in superpixels])
    ys = np.array([sp.posi_y for sp in superpixels])
    zs = np.array([sp.posi_z for sp in superpixels])
    ints = np.array([sp.mean_intensity for sp in superpixels])
    xns = np.array([sp.norm_x for sp in superpixels])
    yns = np.array([sp.norm_y for sp in superpixels])
    zns = np.array([sp.norm_z for sp in superpixels])
    normnorms = np.sqrt(np.square(xns) + np.square(yns) + np.square(zns))
    xns = xns / normnorms / 15
    yns = yns / normnorms / 15
    zns = zns / normnorms / 15
    ax1.scatter(xs, zs, -ys)#, c=ints, cmap='Greys')
    # ax1.quiver(xs, zs, -ys, xns, zns, -yns)

    # # im3 = spExtractor.calculate_spaces()
    # # xs = im3[:,:,0].flatten()
    # # ys = im3[:,:,1].flatten()
    # # zs = im3[:,:,2].flatten()
    # # ax1.scatter(xs, zs, -ys)#, c=imgray.flatten())
    # ax1.set_xlabel('x')
    # ax1.set_ylabel('y')
    # ax1.set_zlabel('z')
    # ax1.set_xlim3d(-0.4, 0.4)
    # ax1.set_ylim3d(-0.4, 0.4)
    # ax1.set_zlim3d(-0.4, 0.4)

    print('done!')

    plt.show()

if __name__ == '__main__':
    main()
