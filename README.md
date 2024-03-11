This repository is an unofficial implementation for BEVPlace: Learning LiDAR-based Place Recognition using Bird's Eye View Images. Significant changes to hyperparameters and details may be necessary.


### Setting up Environment
```
conda create -n bev python=3.8 
conda activate bev 
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 -c pytorch
pip install scikit-learn tensorboardx==2.6.2.2 opencv-python scikit-image

```

### Dataset preparation

Download benchmark_datasets.zip from [here](https://drive.google.com/drive/folders/1Wn1Lvvk0oAkwOUwR0R6apbrekdXAUg7D).

### Preprocessing dataset
'''
python upscale_bin.py 
'''

### For training tuples in our network
```
cd generating_queries/
python generate_training_tuples_baseline.py
```
### For test tuples in our network
```
cd generating_queries/
python generate_test_sets.py
```

### Train

```
python train.py 
```

### Evaluate
```
python evaluate.py 
```

## References:
The code is in built on [BEVPlace](https://github.com/zjuluolun/BEVPlace).

1. Luo, L., Zheng, S., Li, Y., Fan, Y., Yu, B., Cao, S.Y., Li, J., & Shen, H.L., 2023. BEVPlace: Learning LiDAR-based place recognition using bird's eye view images. In *Proceedings of the IEEE/CVF International Conference on Computer Vision* (pp. 8700-8709).
