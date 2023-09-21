# [Iterative Graph Filtering Network for 3D Human Pose Estimation](https://www.sciencedirect.com/science/article/pii/S104732032300158X) (Journal of Visual Communication and Image Representation, 2023)

This repository contains the official PyTorch implementation of the [Iterative Graph Filtering Network for 3D Human Pose Estimation](https://arxiv.org/pdf/2307.16074v2.pdf) authored by Zaedul Islam and A. Ben Hamza. If you discover our code to be valuable for your research, kindly consider including the following citation:

```
@article{islam2023iterative,
  title={Iterative graph filtering network for 3D human pose estimation},
  author={Islam, Zaedul and Hamza, A Ben},
  journal={Journal of Visual Communication and Image Representation},
  volume={95},
  pages={103908},
  year={2023},
  publisher={Elsevier}
}
```

## Network Architecture
<p align="center"><img src="demo/Network_Architecture.png", width="1000" alt="" /></p>


## Qualitative Results
<p align="center"><img src="demo/Photo.gif", width="1000" alt="" /></p>


## Results on Human3.6M
Results under Protocol #1 (mean per-joint position error) and Protocol #2 (mean per-joint position error after rigid alignment).

| Method | MPJPE (P1) | PA-MPJPE (P2) |
|  :----:  | :----: | :----: |
| [SemGCN](https://github.com/garyzhao/SemGCN) | 57.6mm | - |
| [High-order GCN](https://github.com/ZhimingZo/HGCN) | 55.6mm | 43.7mm |
| [HOIF-Net](https://github.com/happyvictor008/Higher-Order-Implicit-Fairing-Networks-for-3D-Human-Pose-Estimation) | 54.8mm | 42.9mm |
| [Weight Unsharing](https://github.com/tamasino52/Any-GCN) | 52.4mm | 41.2mm |
| [MM-GCN](https://github.com/JaeYungLee/MM_GCN) | 51.7mm | 40.3mm |
| [Modulated GCN](https://github.com/ZhimingZo/Modulated-GCN) | 49.4mm | 39.1mm |
| Ours | **47.1mm** | **38.7mm** |

## Quick Start
This repository is built upon Python v3.8 and Pytorch v1.8.0 on Ubuntu 20.04.4 LTS. All experiments are conducted on a single NVIDIA RTX 3070 GPU with 8GB of memory.

## Dependencies
Please make sure you have the following dependencies installed:

* PyTorch >= 1.8.0
* NumPy
* Matplotlib

## Dataset
Our model is evaluated on [Human3.6M](http://vision.imar.ro/human3.6m) and [MPI-INF-3DHP](https://vcai.mpi-inf.mpg.de/3dhp-dataset/) datasets. Please put the datasets in `./dataset` directory.

### Human3.6M 
2D detections for Human3.6M dataset are provided by [VideoPose3D](https://github.com/facebookresearch/VideoPose3D) Pavllo et al.

### MPI-INF-3DHP
We set up the MPI-INF-3DHP dataset in the same way as [PoseAug](https://github.com/jfzhang95/PoseAug). Please refer to [DATASETS.md](https://github.com/jfzhang95/PoseAug/blob/main/DATASETS.md) to prepare the dataset file.

## Training from Scratch
### Human3.6M

To initiate the training of our model, utilizing the identified 2D keypoints (HR-Net) along with pose refinement, please execute the following command:
```bash
python main_graph.py  --pro_train 1 --beta 0.2 --k hrn --batchSize 512 --hid_dim 384 --save_model 1  --save_dir './checkpoint/train_result/' --post_refine --save_out_type post --show_protocol2
```

To initiate the training of our model, utilizing the ground truth 2D keypoints excluding pose refinement and non-local layer, please execute the following command:
```bash
python main_graph.py  --pro_train 1 --beta 0.2 --k gt --batchSize 512 --hid_dim 384 --save_model 1 --save_dir './checkpoint/train_result/' --show_protocol2
```

## Evaluation
### Human3.6M
To evaluate our model using the detected 2D keypoints (HR-Net) with pose refinement, please run the following command:
```bash
python main_graph.py -k hrn --beta 0.2 --batchSize 512 --hid_dim 384 --previous_dir './checkpoint/train_result/' --save_out_type post --save_dir './checkpoint/test_result/' --gsnet_gcn_reload 1 --module_gsnet_model [model_gsnet].pth --post_refine --post_refine_reload 1 --post_refine_model `[model_post_refine]`.pth --show_protocol2 --nepoch 2
```

To evaluate our model using the ground truth 2D keypoints without incorporating pose refinement and and non-local layer, please run the following command:
```bash
python main_graph.py -k gt --beta 0.2 --batchSize 512 --hid_dim 384 --previous_dir './checkpoint/train_result/' --save_dir './checkpoint/test_result/' --save_out_type xyz --gsnet_gcn_reload 1 --module_gsnet_model `[model_gsnet]`.pth --show_protocol2 --nepoch 2
```

## Evaluating Our Pre-trained Models
The pre-trained models can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1ByA9gmEEJSMJetQoIuRA2PYfQ7hVhJ3v?usp=sharing). Put them in the `./checkpoint/train_result` directory.

To evaluate our pre-trained model using the ground truth 2D keypoints, please run:
```bash
python main_graph.py -k gt --beta 0.2 --batchSize 512 --hid_dim 384 --previous_dir './checkpoint/train_result/' --save_dir './checkpoint/test_result/' --save_out_type xyz --gsnet_gcn_reload 1 --module_gsnet_model model_gsnet_gcn_7_eva_xyz_3649.pth --show_protocol2 --nepoch 2
```

## Acknowledgement
Our code makes references to the following repositories.
* [Modulated GCN](https://github.com/ZhimingZo/Modulated-GCN)
* [SemGCN](https://github.com/garyzhao/SemGCN)
* [PoseAug](https://github.com/jfzhang95/PoseAug)
* [VideoPose3D](https://github.com/facebookresearch/VideoPose3D)

We thank the authors for sharing their code and kindly request that you also acknowledge their contributions by citing their work.
