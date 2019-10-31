import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from superpixel_extraction import SuperpixelExtraction


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

def main():

    image_full = plt.imread('../dataset/rgb/kitti.png')
    image_full = gray2rgb(image_full)
    depth_full = plt.imread('../dataset/depth/kitti.png')# / 5000
    depth_full = rgb2gray(depth_full)
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
                                       weights=weights, sp_size=int(25*scale))
    
    # original images
    plt.figure(1, figsize=(18,8))
    plt.subplot(2,1,1)
    plt.imshow(image_full)
    plt.subplot(2,1,2)
    plt.imshow(depth_full)
    plt.draw();plt.pause(0.001)
    # superpixels
    superpixels = spExtractor.init_seeds()
    for i in range(25):
        superpixel_idx = spExtractor.assign_pixels(superpixels)
        superpixels = spExtractor.update_seeds(superpixel_idx, superpixels)
        # plt.subplot(2,1,1)
        image_sp = plot_sp(image_full, superpixel_idx, superpixels, toPlot=False)
        # plt.subplot(2,1,2)
        depth_sp = plot_sp(depth_full, superpixel_idx, superpixels, toPlot=False)
        plt.imsave('../dataset/results/kitti_superpixels_rgb{:02d}'.format(i), image_sp)
        plt.imsave('../dataset/results/kitti_superpixels_depth{:02d}'.format(i), depth_sp)
    # superpixels = spExtractor.calc_norms(superpixel_idx, superpixels)

    # # plotting
    plt.subplot(2,1,1)
    image_sp = plot_sp(image_full, superpixel_idx, superpixels)
    plt.subplot(2,1,2)
    depth_sp = plot_sp(depth_full, superpixel_idx, superpixels)

    plt.imsave('../dataset/results/kitti_superpixels_rgb', image_sp)
    plt.imsave('../dataset/results/kitti_superpixels_depth', depth_sp)

    # plt.figure(2)
    # ax1 = plt.subplot(1,1,1, projection='3d')
    # xs = np.array([sp.posi_x for sp in superpixels])
    # ys = np.array([sp.posi_y for sp in superpixels])
    # zs = np.array([sp.posi_z for sp in superpixels])
    # ints = np.array([sp.mean_intensity for sp in superpixels])
    # xns = np.array([sp.norm_x for sp in superpixels])
    # yns = np.array([sp.norm_y for sp in superpixels])
    # zns = np.array([sp.norm_z for sp in superpixels])
    # normnorms = np.sqrt(np.square(xns) + np.square(yns) + np.square(zns))
    # xns = xns / normnorms / 15
    # yns = yns / normnorms / 15
    # zns = zns / normnorms / 15
    # ax1.scatter(xs, zs, -ys)#, c=ints, cmap='Greys')
    # # ax1.quiver(xs, zs, -ys, xns, zns, -yns)

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
