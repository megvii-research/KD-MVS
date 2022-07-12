<!-- # KD-MVS: Knowledge Distillation Based Self-supervised Learning for Multi-view Stereo -->
# KD-MVS

### [Paper]() | [Project Page]() | [Data](https://drive.google.com/file/d/1aM2_o5kQJbaYVkzrzQhgIpbBkrLSeCYd/view?usp=sharing) | [Checkpoints](https://drive.google.com/drive/folders/1Ctx_zADvjgYpfgtUepkqh31bbs4IotJq?usp=sharing)

<!-- This repo is under constructing. -->

## Installation

Clone this repo:
```
git clone https://github.com/megvii-research/KD-MVS.git
cd KD-MVS
```

We recommend to use [Anaconda](https://www.anaconda.com/) to manage python environment:
```
conda create -n kdmvs python=3.6
conda activate kdmvs
pip install -r requirements.txt
```



##  Data preparation
###  Pseudo label
Download our pseudo label (DTU dataset) from [here](https://drive.google.com/file/d/1aM2_o5kQJbaYVkzrzQhgIpbBkrLSeCYd/view?usp=sharing). Unzip it and put it in the ``CHECKED_DEPTH_DIR`` folder as introduced below.


###  DTU
Download the preprocessed [DTU training data](https://drive.google.com/file/d/1eDjh-_bxKKnEuz5h-HXS7EDJn59clx6V/view)
(from [Original MVSNet](https://github.com/YoYo000/MVSNet)), and unzip it to construct a dataset folder like:
```
dtu_training
 ├── Cameras
 └── Rectified
```
For DTU testing set, download the preprocessed [DTU testing data](https://drive.google.com/open?id=135oKPefcPTsdtLRzoDAQtPpHuoIrpRI_) (from [Original MVSNet](https://github.com/YoYo000/MVSNet)) and unzip it as the test data folder, which should contain one ``cams`` folder, one ``images`` folder and one ``pair.txt`` file.



## Training

### Unsupvised training
Set the configuration in ``scripts/run_train_unsup.sh`` as:
* Set ``DATASET_DIR`` as the path of DTU training set.
* Set ``LOG_DIR`` as the path to save the checkpoints.
* Set ``NGPUS`` and ``BATCH_SIZE`` according to your machine.
* (Optional) Modify other hyper-parameters according to the argparse in ``train_unsup.py``, such as ``summary_freq``, ``save_freq``, and so on.

To train your model, just run:
```
bash scripts/run_train_unsup.sh
```


### KD training
Set the configuration in ``scripts/run_train_kd.sh`` as:
* Set ``DATASET_DIR`` as the path of DTU training set.
* Set ``LOG_DIR`` as the path to save the checkpoints.
* Set ``CHECKED_DEPTH_DIR`` as the path to our pseudo label folder.
* Set ``NGPUS`` and ``BATCH_SIZE`` according to your machine.
* (Optional) Modify other hyper-parameters according to the argparse, such as ``summary_freq``, ``save_freq``, and so on.

To train student model, just run:
```
bash scripts/run_train_kd.sh
```
Note: ``2~3`` or more rounds of KD-training with different thresholds are needed to get fine scores when training from scratch (as shown in Tab. 6), which is somewhat time-consuming.


## Testing
For easy testing, you can download our [pretrained models](https://drive.google.com/drive/folders/1Ctx_zADvjgYpfgtUepkqh31bbs4IotJq?usp=sharing) and put them in `ckpt` folder, or use your own models and follow the instruction below.

Set the configuration in ``scripts/run_test_dtu.sh``:
* Set ``TESTPATH`` as the path of DTU testing set.
* Set ``CKPT_FILE`` as the path of the model weights.
* Set ``OUTDIR`` as the path to save results.
<!-- * Set ``FUSIBILE_EXE`` as the path of [fusibile](https://github.com/kysucix/fusibile). -->

Run:
```
bash scripts/run_test_dtu.sh
```

<!-- The instruction of installing and compiling `fusibile` fusion method can be found [here](https://github.com/kysucix/fusibile). -->

To get quantitative results of the fused point clouds from the official MATLAB evaluation tools, you can refer to [TransMVNet](https://github.com/MegviiRobot/TransMVSNet).



<!--
## Citation
```bibtex

```
-->

## Acknowledgments
We borrow some code from [TransMVSNet](https://github.com/MegviiRobot/TransMVSNet) and [U-MVS](https://github.com/ToughStoneX/U-MVS). We thank the authors for releasing the source code.
