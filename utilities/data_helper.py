"""A data helper module to read data from the dataset folder."""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
from gtsam import Pose3, Rot3, Point3


def read_depth_image(index):
    """Read the dataset/depth/index.png depth image and return an numpy array.
    Return:
        depth_img - height x width x 1 numpy array
    """
    file_name = "dataset/depth/{}.png".format(index)
    depth_image = plt.imread(file_name)
    return depth_image


def read_rgb_image(index):
    """Read the dataset/rgb/index.png rgb image and return an numpy array.
    Return:
        rgb_img - height x width x 3 numpy array
    """
    file_name = "dataset/rgb/{}.png".format(index)
    rgb_image = plt.imread(file_name)
    return rgb_image


def generate_pointcloud(index, focalLength=525.0, centerX=319.5, centerY=239.5, scalingFactor=5000.0):
    """Return a color pointcloud of with a pair of depth image and rgb image.
    Refer from: https://svncvpr.in.tum.de/cvpr-ros-pkg/trunk/rgbd_benchmark/rgbd_benchmark_tools/src/rgbd_benchmark_tools/generate_pointcloud.py
    Return:
        pointcloud - height x width x 6 (x,y,z,r,g,b) numpy array
    """
    rgb_file = "dataset/rgb/{}.png".format(index)
    depth_file = "dataset/depth/{}.png".format(index)
    rgb = Image.open(rgb_file)
    depth = Image.open(depth_file)

    if rgb.size != depth.size:
        raise Exception(
            "Color and depth image do not have the same resolution.")
    if rgb.mode != "RGB":
        raise Exception("Color image is not in RGB format")
    if depth.mode != "I":
        raise Exception("Depth image is not in intensity format")

    points = np.zeros((rgb.size[1], rgb.size[0], 6))
    for v in range(rgb.size[1]):
        for u in range(rgb.size[0]):
            color = rgb.getpixel((u, v))
            Z = depth.getpixel((u, v)) / scalingFactor
            if Z == 0: continue
            X = (u - centerX) * Z / focalLength
            Y = (v - centerY) * Z / focalLength
            points[v][u] = np.array([[X, Y, Z, color[0], color[1], color[2]]])
    return points


def read_ground_truth_poses():
    """Read the dataset/traj3.gt.freiburg file and parse the ground truth into a list of Pose3 items.
    Return:
        rgb_img - height x width x 3 numpy array
    """
    file_path = 'dataset/traj3.gt.freiburg'
    with open(file_path) as file:
        lines = file.readlines()[:]
    def get_pose(line):
        line = list(map(np.float, line.split()))
        x=line[1]
        y=line[2]
        z=line[3]
        qx=line[4]
        qy=line[5]
        qz=line[6]
        qw=line[7]
        rotation = Rot3()
        point = Point3(x,y,z)
        rot = rotation.Quaternion(qw,qx,qy,qz)
        pose3 = Pose3(rot, point)
        return pose3.matrix()

    poses = [get_pose(line) for line in lines]
    return poses

def get_ground_truth_pose(index, poses):
    """Read the dataset/traj3.gt.freiburg file and parse the ground truth into a list of Pose3 items.
    Return:
        translation - 1x3 numpy array
        rotation - 3x3 numpy array
    """
    pass

def F2Pc(point2, depth):
    """Feature to 3D point in the pose coordinate."""
    pass

def F2Pw(point2, depth, pose3):
    """Feature to 3D point in the world coordinate."""
    pass


# depth_image= read_depth_image(0)
# plt.imshow(depth_image)
# plt.show()

# rgb_image= read_rgb_image(0)
# plt.imshow(rgb_image)
# plt.show()

# pointcloud= generate_pointcloud(0)
# shape = pointcloud.shape
read_ground_truth_poses()