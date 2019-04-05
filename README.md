# OM-Net

This is the code release for the MICCAI 2018 paper "One-Pass Multi-task Convolutional Neural Networks for Efficient Brain Tumor Segmentation", authored by Chenhong Zhou, Changxing Ding, Zhentai Lu, Xinchao Wang and Dacheng Tao.


This work overcomes the shortcomings of popular model cascade strategy that can lead to undesired system complexity and ignore the relevance among the models due to its multiple separate cascaded models. We propose to adopt multi-task learning to integrate multiple segmentation tasks into a deep model which is called [*One-pass Multi-task Network (OM-Net)*](https://doi.org/10.1007/978-3-030-00931-1_73).

OM-Net exploits correlation among tasks to achieve superior performance, with only one-third model parameters of model cascade strategy and one-pass computation. It ranks 1st on the BraTS 2015 test set and achieves top performance on BraTS 2017 validation set.


## Model

Firstly, we split multi-class brain tumor segmentation into three segmentation tasks: 1)*Coarse segmentation to detect complete tumor.
2)*Refined segmentation for complete tumor and its intra-tumoral classes*. 3) *Precise segmentation for enhancing tumor*.

For model cascade framework, each task is implemented by an independent basic network. Our basic network used in model cascade framework is a 3D variant of the [Fusionnet](FusionNet: A deep fully residual convolutional neural network for image segmentation in connectomics).See picture below. 

![img/3D_fusionnet.png](img/3D_fusionnet.png)

OM-Net integrates three tasks with their respective training data being the same as those in model cascade framework. Its architecture is illustrated in the picture below. Each task owns its task-specific parameters and the shared backbone model is to learned the correlation among the tasks. Specifically, the shared backbone model refers to the network layers outlined by the yellow dashed line in 3D_fusionnet.png.

![img/om-net-architecture.png](img/om-net-architecture.png)



## Usage

All implementations are based on the [C3D](https://github.com/facebook/C3D), which is a 3D modified version of BVLC caffe.

Please clone the C3D repository and add these files.


To be continued.




## Contact

eezhouch At mail.scut.edu.cn








