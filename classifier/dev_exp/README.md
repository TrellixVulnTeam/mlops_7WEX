# Classification

#### Title
[Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)\
[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

![alt text](./img/paper1.png "Novelty of pix2pix")


# Train CIFAR10 with PyTorch

I'm playing with [PyTorch](http://pytorch.org/) on the CIFAR10 dataset.

## Prerequisites
- Python                 3.7+
- torch                  1.7.1+cu110
- torchvision            0.8.2+cu110
- matplotlib             3.3.3
- numpy                  1.19.5
- Pillow                 8.1.0
- scikit-image           0.17.2
- scipy                  1.5.4
- tensorboard            2.4.1
- tensorboardX           2.1
- tqdm                   4.56.0

## Training

    $ python main.py --mode train \

---

* Set 
* Hyperparameters were written to **arg.txt** under the **[log directory]**.
* To understand hierarchy of directories based on their arguments, see **directories structure** below. 


## Test
    $ python main.py --mode test \

---

* To test using trained network, set **[scope name]** defined in the **train** phase.
* Generated images are saved in the **images** subfolder along with **[result directory]** folder.


## Tensorboard

    $ tensorboard --logdir ./log/* \
                  --port 6006
                  
After the above comment executes, go **http://localhost:6006**

* You can change **[(optional) 4 digit port number]**.
* Default 4 digit port number is **6006**.

## Results
  


## Accuracy
| Model             | Acc.        |
| ----------------- | ----------- |
| [VGG16](https://arxiv.org/abs/1409.1556)              | 92.64%      |
