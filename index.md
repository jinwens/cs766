---
layout: default
---
# TNTU-Net: A Semantic Segmentation Model for Autonomous Driving

## Project Motivation & Introduction

Nowadays, autonomous driving has become increasingly popular and applicable because of the rapid development of the image sensing techniques and computer vision algorithms. 
Among the computer vision methods, semantic segmentation is rather important since it can help incorporate the inferred knowledge (such as the detected object) to enable the decision of autonomous driving. 
It refers to the process of classifying each pixel of an image into semantically similar labels.

In this project, we aim at the construction and evaluation of a semantic segmentation model based on Transformer-iN-Transformer (TNT) and U-Net. 
The proposed model leverages both the TNT and the U-Net structures to achieve a more precise localization as well as a better understanding of global information. 
In addition, different loss functions are considered in the model formulation to improve the classification performance as well as tackle the data imbalance issue.

## Background

Semantic segmentation is an important research direction in the field of computer vision. 
Traditional machine learning and computer vision techniques have been utilized to address such problems in the past, but with the emergence of deep learning, especially Convolutional Neural Network (CNN), the accuracy and efficiency of the approach has increased exponentially. 
The fully convolutional network (FCN) with an encoder-decoder architecture has been the popular paradigm for semantic segmentation. 
However, one weakness of the pure convolution architecture is that the global context is unavoidably not well modeled. 

<p align="center">
<img width="800" src="https://raw.githubusercontent.com/jinwens/cs766/gh-pages/assets/Picture1.png">
</p>

Recently, the new kind of neural architecture transformer, which can provide the relationships between different features based on self-attention mechanism, has been widely promoted as a powerful alternative for computer vision problems. 
Transformer-in-transformer which called TNT is an evolution of transformer. Typically, TNT are doing the same process as transformer but do it twice to get more powerful local features.
Specifically, Han et. al proposed the Transformer-iN-Transformer network architecture which takes into account the attention inside the local patches of images and achieved better accuracy on the ImageNet benchmark. 

<p align="center">
<img width="800" src="https://raw.githubusercontent.com/jinwens/cs766/gh-pages/assets/Picture3.png">
</p>

Besides, the U-Net architecture, which decodes that up-samples features using transposed convolution corresponding to each downsampling stage, presented good performance for medical image segmentation tasks. The symmetric expanding path uses CNN module and up convolutional module to do the up sampling until the features become almost the same size as input. It is powerful to capture global context information.

## Model Formulation

Because of the limitations of previous models, in this project, we propose a semantic Segmentation Model named TNTU-Net, which combines both features of TNT and U-Net.
We replaced CNN module in the U-net as TNT module to create a new semantic segmentation. 

<p align="center">
<img width="500" src="https://raw.githubusercontent.com/cmilica/cs766project/gh-pages/assets/hallway.png">
</p>

On the left side of the figure above, The gray arrow indicates TNT module which is a feature extractor. we also copied the features on the left to the right by using CNN module. And then up sample the features to the same size as the input. 

In order to evaluate the relative performance of reinforcement learning with and without the inclusion of SLAM-derived features, 
we initially trained an agent to complete a modified version of the MiniWorld-Hallway-v0 environment.

<p align="center">
<img width="500" src="https://raw.githubusercontent.com/jinwens/cs766/gh-pages/assets/Picture6.png">
</p>


### Loss Functions

In our model, three types of loss function are utilized to explore their influence on the model performance. 
The following loss functions like binary cross entropy loss, focal loss and dice loss are commonly used in the semantic segmentation task. 
Also we tried the combinations of different loss functions. For example, focal loss + dice loss and binary cross entropy loss + dice loss. 


## Dataset

The data set we used is from KITTI, which is a semantic segmentation benchmark. 
It consists of 200 semantically annotated train as well as 200 test images. 
And there are 11 categories/labels for the image, including building, tree, sky, car, sign, road, pedestrian, fence, pole, sidewalk, and bicyclist.
We implemented experiments on this benchmark dataset to demonstrate the effectiveness of the proposed TNTU-Net architecture.

## Model Evaluation

### TNTU-Net Configuration for Training

### Evaluation Metric

mIOU is a common evaluation metric for semantic image segmentation, which first computes the IOU for each semantic class and then computes the average over classes. 
IOU = true_positive / (true_positive + false_positive + false_negative)

### Evaluation Results

At first, we trained the model with BCE loss, Focal loss, and Dice Loss function. the model with dice loss had the best performance. Then, we tried to improve the model performance. there is one paper suggesting us using focal loss + dice loss. but its performance is even worse. 
After we observed the results, it turns out the predictions of the model with BCE loss function have less noises than other functions. And the model with dice loss had the best performance so far. Thus, we combine BCE loss function and dice loss to train the final model. In the end, we get the best model performance with mean IOU of 61.23.



It's worth mentioning that most SLAM library are designed for ROS and C++ platform, such as orb-SLAM2 and lsd-SLAM, therefore we spent great effort in letting SLAM running on a python simulation environment without ROS.

- Modify step() function where each step action is defined in gym-miniworld so that it will call orb-slam2 directly
- Modify the python-binding for processMono() so that it will directly output 4x4 homogenous transformation matrix which contain position and orientation.
- Interpolate each action into 10 frames and feed into SLAM
- Camera calibration:
   - OpenGL has some different way of defining the projection matrix and use angle of view to represent intrinsic camera information.
- Focal length:
- Alpha = 2 arctan(d/2f)

    where d is the dimension of image, alpha is the angle of view, f is the focus length.
- Principal points: 
   - camera set by gluperspective() function requires zero principle when map into normalized device coordinates(NDC), but since we export the rendering and fed that into SLAM as a OpenCV image, the principal points then becomes the center of image

<p align="center">
<iframe width="560" height="315" src="https://www.youtube.com/embed/M9-AGSCgOJ0" tframeborder="0" allow="accelerometer; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</p>

### PySLAM

In addition to Orb-SLAM 2, we used a modified version of pyslam to extract position and orientation. 
We reconfigured pyslam to accept an image as an argument and return the position and orientation quaternion. 
Pyslam also provides a monocular SLAM implementation, so it forms its localization and mapping estimates from a sequence of single-camera RGB images

<p align="center">
<iframe width="560" height="315" src="https://www.youtube.com/embed/Ofq9iDRJG6I" frameborder="0" allow="accelerometer; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</p>


### ORB-SLAM3

ORB-SLAM3 is the latest in the ORB-SLAM libraries. 
There was a possibility that certain disadvantages of the other two libraries, like speed, and sudden map disruptions,
would be resolved with ORB-SLAM3. 
After multiple attempts to make this work, and trying to modify some of the python bindings,
we were unable to implement this approach. 

However, we did do some background research how this library would work, so we provide a brief summary of some novelties and advantages.

<p align="center">
<img width="300" src="https://raw.githubusercontent.com/cmilica/cs766project/gh-pages/assets/ORB-SLAM3.png">
</p>

Atlas is a multi-map representation composed of a set of disconnected maps.  
The tracking threadtakes input from the correct frame with respect to the active map in Atlas in real-time, 
and it decided whether the currentframe  becomes  a  keyframe.   
If  tracking  is  lost,  the  thread  tries  to  relocate  the  current  thread.   
Depending  on  that,  it decides whether the tracking is resumed, switching the active map if needed, 
or whether or not the active map is storedas non-active,  and a new map is initiated as an active map.  
The next component is a local mapping thread that adds keyframes and points to the active map, removes the redundant ones, 
and refines the map using visual or visual-inertialbundle adjustment. 
Also, in the inertial case, the IMU parameters are initialized and refined by the mapping thread usingthe MAP-estimation technique.  
The last component is the loop and map merging thread that detects common regions between the active map and the whole Atlas at keyframe rate.  
The way this works is that the system checks whetherthe common area belongs to the active map.  
The system then performs loop correction; if it belongs to a different map,both maps are seamlessly merged into a single one, 
and this becomes a new active map.  
Finally, a full BA runs as anindependent thread to prevent anything affecting a real-time map performance.

## Results

### ORB-SLAM2

Here is a demo video that pytorch is trying the agent with SLAM results in real-time. 

<p align="center">
<iframe width="560" height="315" src="https://www.youtube.com/embed/iari7YP6ovI" frameborder="0" allow="accelerometer; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</p>

Timing issue:

| Settings                       | Time per action | 
|:-------------------------------|:----------------|
| 10frames/action​ With viewer    | 0.34s           | 
| 10frames/action​ Without viewer | 0.33s           | 
| 20frames/action                | 0.67s           |
| 5frames/action                 | 0.17s           | 


Resolution: 800x600

Problems to be solved:​

1) speed, SLAM module will slow down the training speed​

2) unexpected loop-closure & lost track: richness of texture inside simulation environment


### PySLAM

We created a set of 200 reference images using our agent trained through standard DQN in modified MiniWorld-Hallway-v0 environment. 
We analyzed the time it took to perform SLAM on this sequence of images in pyslam


## Future work

1) Encode the slam results into DQN model input

2) Improve the code performance to get faster performance

3) Find appropriate parameters that mitigate the "lost of track" issue


## Project Proposal

[Link to the project proposal](./assets/766_final_project.pdf)

## Midterm Report 

[Link to the midterm report](./assets/766_midterm_report.pdf)

## Project code repo

https://github.com/existentmember7/TNTUNet

## Project Timeline

| When                 | Task                                               | 
|:---------------------|:---------------------------------------------------|
| Before Feb 24        | Project Proposal and the initial webpage           | 
| Feb 25 - Mar 10      | Create DQN Miniworld benchmark                     | 
| Mar 11- Mar 20       | Set up SLAM with Miniworld                         |
| Mar 21- Apr 6        | Design input encoding for SLAM features into       | 
| Apr 7 - Apr 21       | Contrast performance of DQN on Miniworld           | 
| Before May 5         | Complete project writeup and presentation          | 



## Welcome to GitHub Pages

You can use the [editor on GitHub](https://github.com/jinwens/cs766/edit/gh-pages/index.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [Basic writing and formatting syntax](https://docs.github.com/en/github/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/jinwens/cs766/settings/pages). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.
