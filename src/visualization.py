import cv2
import matplotlib.pyplot as plt
import numpy as np

from superpixel_extraction import SuperpixelExtraction


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def main():

    image = plt.imread('../dataset/rgb/0.png')
    depth = plt.imread('../dataset/depth/0.png')
    image = cv2.resize(image, (48*3, 64*3))
    depth = cv2.resize(depth, (48*3, 64*3))
    imgray = rgb2gray(image)
    (h,w) = imgray.shape
    camera_parameters = {'fx': 1, 'fy': 1, 'cx': w/2, 'cy': h/2}
    weights = {'Ns': 2000, 'Nc': 5, 'Nd': 5}
    spExtractor = SuperpixelExtraction(imgray, depth, camera_parameters,
                                       weights=weights, sp_size=8)
    
    print("hi")
    # original images
    plt.figure(1)
    plt.subplot(1,2,1)
    plt.imshow(image)
    plt.subplot(1,2,2)
    plt.imshow(depth)
    plt.draw();plt.pause(0.001)
    # sp
    def plot_sp(superpixel_idx, superpixels):
        plt.subplot(1,2,1)
        imnew = np.zeros(imgray.shape, dtype=bool)
        imnew[:-1, :] = (np.diff(superpixel_idx, axis=0) != 0)
        imnew[:, :-1] = (np.diff(superpixel_idx, axis=1) != 0) | imnew[:, :-1]
        imagesp = np.copy(image)
        imagesp[imnew,0] = 1
        imagesp[imnew,1] = 0
        imagesp[imnew,2] = 0
        plt.imshow(imagesp)
        plt.scatter([sp.x for sp in superpixels], [sp.y for sp in superpixels], s=1)
        plt.draw();plt.pause(0.001)
    superpixels = spExtractor.init_seeds()
    for _ in range(15):
        superpixel_idx = spExtractor.assign_pixels(superpixels)
        superpixels = spExtractor.update_seeds(superpixel_idx, superpixels)
        # plot_sp(superpixel_idx, superpixels)
    
    plot_sp(superpixel_idx, superpixels)

    # norm update

    plt.show()

if __name__ == '__main__':
    main()
