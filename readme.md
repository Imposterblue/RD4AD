﻿## CVPR2022 - Anomaly Detection via Reverse Distillation from One-Class Embedding
 ## Implementation (Official Code ⭐️ ⭐️ ⭐️ )

1. Environment
	> torch == 1.9.1
	
	> torchvision == 0.10.1
	
	> numpy == 1.20.3
	
	> scipy == 1.7.1
	
	> sklearn == 1.0
	
	> PIL == 8.3.2

```
torch
torchvision == 0.10.1
opencv-python
matplotlib
scikit-image
pandas
numpy == 1.20.3
scipy == 1.7.1
scikit-learn == 1.0
Pillow == 8.3.2
```

2. Dataset
    > You should download MVTec from [MVTec AD: MVTec Software](https://www.mvtec.com/company/research/datasets/mvtec-ad/). The folder "mvtec" should be unpacked into the code folder.
3. Train and Test the Model
We have write both training and evaluation function in the main.py, execute the following command to see the training and evaluation results.
    > python3 main.py
    
 ## Reference
	@InProceedings{Deng_2022_CVPR,
    author    = {Deng, Hanqiu and Li, Xingyu},
    title     = {Anomaly Detection via Reverse Distillation From One-Class Embedding},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {9737-9746}}
