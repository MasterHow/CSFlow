# CSFlow: Learning Optical Flow via Cross Strip Correlation for Autonomous Driving
The implementations of [CSFlow: Learning Optical Flow via Cross Strip Correlation for Autonomous Driving](https://arxiv.org/pdf/2202.00909.pdf). 
We achieve state-of-the-art accuracy on KITTI-2015 flow benchmark.

![](results/compare.png)

# Install
```
python setup.py develop
```

# Pretrained Model
The pretrained model that the paper used can be found there:
```
https://share.weiyun.com/5t6TadPB
```

# Train and Eval
To train, use the following command format:
```
python ./tools/train.py
--model CSFlow
--dataset Chairs
--data_root $YOUR_DATA_PATH$
--batch_size 1
--name csflow-test
--validation Sintel
--val_Sintel_root $YOUR_DATA_PATH$
--num_steps 100
--lr 0.0004
--image_size 368 496
--wdecay 0.0001
```
To eval, use the following command format:
```
python ./tools/eval.py
--model CSFlow
--restore_ckpt ./checkpoints/CSFlow-kitti.pth
--eval_iters 24
--validation KITTI
--val_KITTI_root $YOUR_DATA_PATH$
```
For more details, please check the code or refer our [paper](https://arxiv.org/pdf/2202.00909.pdf).

# Folder Hierarchy
\* local: you should create this folder in your local repository and these folders will not upload to remote repository.
```
├── data (local)            # Store test/training data
├── checkpoints (local)     # Store the checkpoints
├── runs (local)            # Store the training log
├── opticalflow             # All source code
|   ├─ api                  # Called by tools
|   ├─ core                 # Core code call by other directorys. Provide dataset, models ....
|   |   ├─ dataset          # I/O of each dataset
|   |   ├─ model            # Models, includeing all the modules that derive nn.module
|   |   ├─ util             # Utility functions
├── tools                   # Scripts for test and train
├── work_dirs (local)       # For developers to save thier own codes and assets
```

# Devs
Hao Shi，YiFan Zhou

# Need Help?
If you have any questions, welcome to e-mail me: haoshi@zju.edu.cn, and I will try my best to help you. =)