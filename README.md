# SSBDM

## Overview
The implementation of the paper "SSBDM: A Spectral–Spatial Bilinear Decomposition Model with Adaptive Multi-Kernel Dictionary for Hyperspectral Target Detection"

## Requirements
Python 3.10  

scikit-learn 1.5.0

numpy 1.26.4

Matlab R2020b

## Dataset
San Diego I,  San Diego II,  Cuprite,  Xuzhou

Due to the memory limitation, the Cuprite and Xuzhou datasets can be found in https://pan.baidu.com/s/1GzYOl1Y2P9p-_hlu8OWDdw?pwd=3fx6 提取码: 3fx6 

## Run
You can directly run "main_sandiego.m" to test on the San Diego I dataset.

For a new dataset, you should run the "amk-atgp.py" to obtain the background dictionary at first.

## Citation
If you use this code, please cite:
```bibtex
@ARTICLE{11422343,
  author={Shen, Dunbin and Kong, Wenfeng and Xiao, Xiwen and Liu, Jianjun and Du, Zhenrong and Ma, Xiaorui and Zhao, Wenda and Wang, Hongyu},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing}, 
  title={SSBDM: A Spectral–Spatial Bilinear Decomposition Model With Adaptive Multikernel Dictionary for Hyperspectral Target Detection}, 
  year={2026},
  volume={19},
  number={},
  pages={9347-9365},
  keywords={Kernel;Adaptation models;Feature extraction;Context modeling;Dictionaries;Detectors;Robustness;Object detection;Hyperspectral imaging;Atoms;Bilinear mixing;hyperspectral image (HSI);multikernel learning;spectral–spatial regularization;target detection},
  doi={10.1109/JSTARS.2026.3670876}}
```
