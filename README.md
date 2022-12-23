## KD-MVS: Knowledge Distillation Based Self-supervised Learning for Multi-view Stereo

Code for paper [KD-MVS: Knowledge Distillation Based Self-supervised Learning for Multi-view Stereo](https://arxiv.org/abs/2207.10425)

**Tips**: If you meet any problems when reproduce our results, please contact Yikang Ding (dyk20@mails.tsinghua.edu.cn). We are happy to help you solve the problems and share our experience.

## Change log
* 12.2022: Update code and README (e.g., more instruction of training and testing, update the fuse code and the pretrained model).
* 09.2022: Update code (e.g., for cross view check and prob encoding, training scripts).

## Installation

Clone this repo:
```
git clone https://github.com/megvii-research/KD-MVS.git
cd KD-MVS
```

We recommend using [Anaconda](https://www.anaconda.com/) to manage python environment:
```
conda create -n kdmvs python=3.6
conda activate kdmvs
pip install -r requirements.txt
```
We also recommend using `apex`, you can install apex from the [official repo](https://www.github.com/nvidia/apex).


##  Data preparation


###  Training data
Download the preprocessed [DTU training data](https://drive.google.com/file/d/1eDjh-_bxKKnEuz5h-HXS7EDJn59clx6V/view)
(from [Original MVSNet](https://github.com/YoYo000/MVSNet)), and unzip it to construct a dataset folder like:
```
dtu_training
 ├── Cameras
 └── Rectified
```

###  Testing data
Download our processed [DTU testing data](https://drive.google.com/file/d/1PdY1KVBo9nmk6SyFUqx4UVEeO218Nspk/view?usp=share_link) and unzip it as the test data folder, which should contain one ``cams`` folder, one ``images`` folder and one ``pair.txt`` file.



## Training

### Unsupervised training
Set the configuration in ``scripts/run_train_unsup.sh`` as:
* Set ``DATASET_DIR`` as the path of DTU training set.
* Set ``LOG_DIR`` as the path to save the checkpoints.
* Set ``NGPUS`` and ``BATCH_SIZE`` according to your machine.
* (Optional) Modify other hyper-parameters according to the argparse in ``train_unsup.py``, such as ``summary_freq``, ``save_freq``, and so on.

To train your model, run:
```
bash scripts/run_train_unsup.sh
```


### KD training
Note:
* We use the `apex` and `sync_bn` by default, to use these modules, make sure you have installed the `apex` according to the [official repo](https://www.github.com/nvidia/apex).
* We use the `gipuma` fusion method by default, please make sure you have compiled and installed it correctly. To do so, you need clone the modified version from [Yao Yao](https://github.com/YoYo000/fusibile).
Modify the `line-10` in `CMakeLists.txt` to suit your GPUs. Then install it by `cmake .` and `make`, which will generate the executable at `FUSIBILE_EXE_PATH`.
* The number and type of GPUs (as well as batchsize) used in training phase might affect the final results. Using `sync_bn` could help with this problem.

To reproduce the results, please note:
* The checkpoint of the last epoch isn't always the best one. In supervised mode, we can easily pick the best model with the help of `validation set`. However, in self-supervised mode, we need to pick models manually based on experience.
* The performance of different epochs varies greatly, using `apex` and `sync_bn` could be helpful (but still can't handle this problem completely).
* The exprimental results reported in our paper are obtained by using different thresholds in different rounds. However, we also repoduce the results by using the latest code and the same thresholds in different rounds.


Before start training, set the configuration in ``scripts/run_train_kd.sh`` as:
* Set ``DATASET_DIR`` as the path of DTU training set.
* Set ``LOG_DIR`` as the path to save the checkpoints.
* Set ``CHECKED_DEPTH_DIR`` as the path to our pseudo label folder.
* Set ``NGPUS`` and ``BATCH_SIZE`` according to your machine.
* (Optional) Modify other hyper-parameters according to the argparse, such as ``summary_freq``, ``save_freq``, and so on.

run:
```
bash scripts/run_train_kd.sh
```



## Testing
For easy testing, you can download our [pretrained models](https://drive.google.com/drive/folders/1Ctx_zADvjgYpfgtUepkqh31bbs4IotJq?usp=share_link) and put them in `ckpt` folder, or use your own models and follow the instruction below.

Make sure:
* Download [the processed test data](https://drive.google.com/file/d/1PdY1KVBo9nmk6SyFUqx4UVEeO218Nspk/view?usp=share_link).
* Install the fusible `gipuma` package.

Set the configuration in ``scripts/run_test_dtu.sh``:
* Set ``TESTPATH`` as the path of DTU testing set.
* Set ``CKPT_FILE`` as the path of the model weights.
* Set ``OUTDIR`` as the path to save results.
* Set ``FUSIBILE_EXE`` as the path to gipuma fusible file.


Run:
```
bash scripts/run_test_dtu.sh
```
The reconstructed point cloud results would be stored in `outputs/test_dtu/gipuma_pcd`, you can also download our fused point cloud results of KD-trained model from [here](https://drive.google.com/file/d/1OWO8a5Cxv5NamY2_dLfIOp_IuJZi_9iQ/view?usp=share_link).

To get quantitative results of the fused point clouds from the official MATLAB evaluation tools, you can refer to [TransMVSNet](https://github.com/MegviiRobot/TransMVSNet).

By using the latest code, pretrained model and default parameters, you can get the final results like:
|   Model   |   Acc.    | Comp. | Overall |
| --- | --- | --- | --- |
|   unsup   |   0.4166   | 0.4335 | 0.4251 |
|   KD      |   0.3674   | 0.2847 | 0.3260 |




## Citation
```bibtex
@inproceedings{ding2022kdmvs,
  title={KD-MVS: Knowledge Distillation Based Self-supervised Learning for Multi-view Stereo},
  author={Ding, Yikang and Zhu, Qingtian and Liu, Xiangyue and Yuan, Wentao and Zhang, Haotian  and Zhang, Chi},
  booktitle={European Conference on Computer Vision},
  year={2022},
  organization={Springer}
}
```

## Acknowledgments
We borrow some code from [CasMVSNet](https://github.com/alibaba/cascade-stereo/tree/master/CasMVSNet) and [U-MVS](https://github.com/ToughStoneX/U-MVS). We thank the authors for releasing the source code.
