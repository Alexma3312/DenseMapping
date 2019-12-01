---
bibliography: [related_work.bib]
csl: [ieee.csl]
css: [main.css]
link-citations: true
---

# CS6476 Final Project
## Dense Mapping using Feature Matching and Superpixel Clustering

<p style="color:#808080;margin:16px 0 4px 0;font-size:20px">
Mandy Xie, Shicong Ma, Gerry Chen
</p>
<p style="color:#404040;margin:0 0 16px 0">
October 31, 2019
</p>

<hr />

### Abstract
<!-- One or two sentences on the motivation behind the problem you are solving. One or two sentences describing the approach you took. One or two sentences on the main result you obtained. -->
One of the fundamental tasks for robot autonomous navigation is to perceive and
digitalize the surrounding 3D environment [@handa2014benchmark]. We replicate
the results of [@Wang19icra_surfelDense] to produce semi-dense, surfel-based
reconstruction using superpixels.

<!-- ### Teaser figure -->
<!-- A figure that conveys the main idea behind the project or the main application being addressed. -->
![superpixel annotation on RGB
image](./results/superpixels/kitti_superpixels_rgb.gif){width=100%}
![superpixel annotation on depth
image](./results/superpixels/kitti_superpixels_depth.gif){width=100%}

### Introduction
<!-- Motivation behind the problem you are solving, what applications it has, any brief background on the particular domain you are working in (if not regular RBG photographs), etc. If you are using a new way to solve an existing problem, briefly mention and describe the existing approaches and tell us how your approach is new. -->
One of the fundamental tasks for robot autonomous navigation is to perceive and
digitalize the surrounding 3D environment [@handa2014benchmark]. To be usable
in mobile robot applications, the mapping system needs to fast and densely
recover the environment in order to provide sufficient information for
navigation.

Unlike other 3d reconstruction methods that reconstructs the environment as a 3D
point cloud, we hope to extract surfels \cite{schops2018surfelmeshing,
pfister2000surfels, tobor2000rendering} based on extracted superpixels from
intensity and depth images and construct a surfel cloud. This approach is
introduced by [@Wang19icra_surfelDense] which can greatly reduces the memory
burden of mapping system when applied to large-scale missions. More importantly,
outliers and noise from low-quality depth maps can be reduced based on extracted
superpixels.

The **goal** of our project is to reproduce results of Wang et al's, namely
implementing superpixel extraction, surfel initialization, and surfel fusion to
generate a surfel-based reconstruction given a camera poses from a sparse SLAM
implementation.  The **input** to our system is an RGB-D video stream with
accompanying camera poses and the **output** is a surfel cloud map of the
environment, similar to Figures 4b or 8 of the original paper
[@Wang19icra_surfelDense].

### Approach
The idea behind dense mapping is to first generate frame related poses, then
reconstruct the dense map based on pre-generated poses and surfels.

1. Select a RGB-D dataset [@handa2014benchmark; @sturm12iros_TUM; @Menze2015CVPR_KITTI]

2. Read pose information from the dataset / Use a sparse SLAM system (VINS [@qin2018vins]/ORB-SLAM2 [@mur2017orb]) to
estimate camera poses

3. Run code from [@Wang19github] directly to confirm functionality and set benchmark/expectations.

3. **(Suggested implementation)** -- Single frame Superpixels extraction from RGB-D images using a k-means approach adapted from SLIC [@achanta2012slic] - IV.D section in [@Wang19icra_surfelDense]

4. **(Suggested implementation)** -- Single frame surfel generation based on extracted superpixels. - IV.E section in [@Wang19icra_surfelDense]

5. **(Suggested implementation)** --  Surfel fusion and Surfel Cloud update. - IV.G section in [@Wang19icra_surfelDense]

<!-- 6. 3D mesh with surfel cloud. -->

### Experiments and results
<!-- Provide details about the experimental set up (number of images/videos, number of datasets you experimented with, train/test split if you used machine learning algorithms, etc.). Describe the evaluation metrics you used to evaluate how well your approach is working. Include clear figures and tables, as well as illustrative qualitative examples if appropriate. Be sure to include obvious baselines to see if your approach is doing better than a naive approach (e.g. for classification accuracy, how well would a classifier do that made random decisions?). Also discuss any parameters of your algorithms, and tell us how you set the values of those parameters. You can also show us how the performance varies as you change those parameter values. Be sure to discuss any trends you see in your results, and explain why these trends make sense. Are the results as expected? Why? -->
#### Dataset
We have started with the _kt3_ sequence of the ICL-NIUM dataset [@handa2014benchmark].  Images and
depth maps have been extracted and examples shown below.

![rgb image from ICL-NIUM dataset](./results/superpixels/icl_rgb0.png){width=45%}
![depth image from ICL-NIUM dataset](./results/superpixels/icl_depth0.png){width=45%}

rgb image from ICL-NIUM dataset     |  depth image from ICL-NIUM dataset
:-------------------------:|:-------------------------:
<img align="center" src="./results/superpixels/icl_rgb0.png" width="500"/> | <img align="center" src="./results/superpixels/icl_depth0.png" width="500"/>

#### Run Existing Code
The code written for the paper was run to ensure that the results could be
reproduced.  Below are some results of running the code for dense reconstruction on images from the KITTI
dataset [@Menze2015CVPR_KITTI].  We showed that it can indeed produce dense reconstructions.

![kitti](./results/kitti/kitti.png){width=45%}
![kitti](./results/kitti/kitti_front.png){width=45%}

![kitti](./results/kitti/kitti_side.png){width=45%}
![kitti](./results/kitti/kitti_top.png){width=45%}

#### Superpixel Extraction
We have completed single-frame superpixel generation.  The results are shown
below.

![superpixel annotation on RGB image](./results/superpixels/superpixels_rgb.gif){width=45%}
![superpixel annotation on depth
image](./results/superpixels/superpixels_depth.gif){width=45%}

We follow the standard implementation as described in the paper:
1. Initialize superpixel seeds - 
    Superpixel seeds are initialized on a grid of predefined size
2. Update superpixels
    1. Pixels are assigned to their nearest superpixel
    2. Superpixel properties (x, y, size, intensity, depth) are updated
       accordingly.

The number of times that step 2 is repeated depends on the image.  For example,
the ICL-NIUM dataset image requires only 10 iterations or so to stabilize, but the
KITTI dataset image requires roughly 25 to stabilize.  One metric that can
potentially be used in the future as a stopping criteria is the sum of distances
traveled by the superpixels from one iteration to the next.  The superpixel
iteration is complete when they sufficiently small changes occur from one
iteration to the next.

#### Surfel Generation
<!-- norm calculation -->
We are in the process of calculating the norm which is needed for surfel
generation.  We expect to complete this very soon.

The difficulties are 
1. Using matrix manipulation with numpy instead of for loop and multi-threads in C++ to reduce computationl cost.
2. To understand the meaning of different values in a surfel vector. 

### Qualitative results
<!-- Show several visual examples of inputs/outputs of your system (success cases and failures) that help us better understand your approach. -->
The images below demonstrate that the superpixels are indeed segmenting properly
as they tend to "hug" similarly colored/depthed regions.

![superpixel annotation on RGB image](./results/superpixels/kitti_superpixels_rgb.png){width=100%}
![superpixel annotation on depth
image](./results/superpixels/kitti_superpixels_depth.png){width=100%}

### Conclusion and future work
<!-- Conclusion would likely make the same points as the abstract. Discuss any future ideas you have to make your approach better. -->
We will recreate the results of [@Wang19icra_surfelDense] by creating a surfel
cloud given RGBD images and camera poses.

### References
<!-- List out all the references you have used for your work -->