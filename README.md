Source code for the ICCV-2021 paper "Collaborative and Adversarial Learning of Focused and Dispersive Representations for Semi-supervised Polyp Segmentation".
## Abstract

Automatic polyp segmentation from colonoscopy images is an essential step in computer aided diagnosis for colorectal cancer. Most of polyp segmentation methods reported
in recent years are based on fully supervised deep learning. However, annotation for polyp images by physicians during the diagnosis is time-consuming and costly. In this paper, we present a novel semi-supervised polyp segmentation via collaborative and adversarial learning of focused and dispersive representations learning model, where focused and dispersive extraction module are used to deal with the diversity of location and shape of polyps. In addition, confidence maps produced by a discriminator in an adversarial training framework shows the effectiveness of leveraging unlabeled data and improving the performance of segmentation network. Consistent regularization is further employed to optimize the segmentation networks to strengthen the representation of the outputs of focused and dispersive extraction module. We also propose an auxiliary adversarial learning method to better leverage unlabeled examples to further improve semantic segmentation accuracy. We conduct extensive experiments on two famous polyp datasets: Kvasir-SEG and CVC-Clinic DB. Experimental results demonstrate the effectiveness of the proposed model, consistently outperforming state-of-the-art semi-supervised segmentation models based on adversarial training and even some advanced fully supervised models.

## Requirements
* CUDA/CUDNN
* pytorch >= 0.4

## Citation

```
@inproceedings{wu2021collaborative,
  title={Collaborative and Adversarial Learning of Focused and Dispersive Representations for Semi-Supervised Polyp Segmentation},
  author={Wu, Huisi and Chen, Guilian and Wen, Zhenkun and Qin, Jing},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={3489--3498},
  year={2021}
}
```
