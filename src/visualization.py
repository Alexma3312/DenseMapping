import cv2
import matplotlib.pyplot as plt
import numpy as np

from superpixel_extraction import SuperpixelExtraction


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def plot_sp(image, superpixel_idx, superpixels):
    sp_idx_full = cv2.resize(superpixel_idx.astype(np.float), image.shape[1::-1],
                                interpolation=cv2.INTER_NEAREST)
    sp_edges = np.zeros(sp_idx_full.shape, dtype=bool)
    sp_edges[:-1, :] = (np.diff(sp_idx_full, axis=0) != 0)
    sp_edges[:, :-1] = (np.diff(sp_idx_full, axis=1) != 0) | sp_edges[:, :-1]
    image_sp = np.copy(image)
    if (len(image.shape)==3): # rgb
        image_sp[sp_edges,0] = 1
        image_sp[sp_edges,1] = 0
        image_sp[sp_edges,2] = 0
    else: # depth/grayscale
        image_sp[sp_edges] = np.min(image_sp)
    plt.imshow(image_sp)
    plt.scatter(
        [sp.x*image.shape[0]/superpixel_idx.shape[0] for sp in superpixels],
        [sp.y*image.shape[0]/superpixel_idx.shape[0] for sp in superpixels], s=1)
    plt.draw();plt.pause(0.001)
    return image_sp

def main():

    image_full = plt.imread('../dataset/rgb/0.png')
    depth_full = plt.imread('../dataset/depth/0.png')
    image = cv2.resize(image_full, (64*3, 48*3))
    depth = cv2.resize(depth_full, (64*3, 48*3))
    imgray = rgb2gray(image)
    (h,w) = imgray.shape
    camera_parameters = {'fx': 1, 'fy': 1, 'cx': w/2, 'cy': h/2}
    weights = {'Ns': 500, 'Nc': 5, 'Nd': 5}
    spExtractor = SuperpixelExtraction(imgray, depth, camera_parameters,
                                       weights=weights, sp_size=8)
    
    # original images
    plt.figure(1)
    plt.subplot(1,2,1)
    plt.imshow(image_full)
    plt.subplot(1,2,2)
    plt.imshow(depth_full)
    plt.draw();plt.pause(0.001)
    # superpixels
    superpixels = spExtractor.init_seeds()
    for _ in range(15):
        superpixel_idx = spExtractor.assign_pixels(superpixels)
        superpixels = spExtractor.update_seeds(superpixel_idx, superpixels)
    # superpixels = spExtractor.calc_norms(superpixel_idx, superpixels)

    # plotting
    plt.subplot(1,2,1)
    image_sp = plot_sp(image_full, superpixel_idx, superpixels)
    plt.subplot(1,2,2)
    depth_sp = plot_sp(depth_full, superpixel_idx, superpixels)
    plt.imsave('../dataset/results/superpixels_rgb', image_sp)
    plt.imsave('../dataset/results/superpixels_depth', depth_sp)

    print('done!')

    plt.show()

if __name__ == '__main__':
    main()
