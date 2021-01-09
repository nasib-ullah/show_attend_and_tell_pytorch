# Image Captioning : Show attend and tell pytorch
This repository contains Pytorch implementation of the image captioning model published in the paper [Show attend and tell](https://https://arxiv.org/abs/1502.03044) (Xu et al, 2015)


## Environment
* Ubuntu 18.04
* CUDA 11.0
* cuDNN 
* Nvidia GeForce RTX 2080Ti

## Requirements 
* Java 8 
* Python 3.8.5
    * Pytorch 1.7.0
    * Other Python libraries specified in requirements.txt
    
 ## How to Use
 
### Step 1. Setup python virtual environment

```
$ virtualenv .env
$ source .env/bin/activate
(.env) $ pip install --upgrade pip
(.env) $ pip install -r requirements.txt
```
### Step 2. Prepare data and path
```
```
### Step 3. Training
Run
   ```
   (.env) $ python train.py
   ```

You can change some hyperparameters by modifying `config.py`.

### Step 4. Inference
```
```
### Step 5. Prepare Evaluation Codes
```
```
## Quantitative Results

 | Encoder | Trained on | BLEU4 | CIDEr | METEOR | ROUGE_L |
 | :---: | :---: | :---: | :---: | :---: | :---: |
 | VGG | COCO2014 | 41.92 | 51.67 | 29.6 | - |
 | Resnet101 | COCO2014 | 45.3 | 76.2 | 31.9 | 64.2 |

## Qualitative Results

#### Training data
##### (1) Generated Caption : A Train traveling down tracks next to a Forest.
![](https://github.com/nasib104/show_attend_and_tell_pytorch/blob/main/images/train_1.png "Logo Title Text 1")

##### (2) Generated Caption : A man riding a Skateboard down a street.
![](https://github.com/nasib104/show_attend_and_tell_pytorch/blob/main/images/train_2.png "Logo Title Text 1")

#### Validation data
##### (1) Generated Caption : A group of people standing around a truck.
![](https://github.com/nasib104/show_attend_and_tell_pytorch/blob/main/images/val_1.png "Logo Title Text 1")

##### (2) Generated Caption : A dog sitting on a boat in the water.
![](https://github.com/nasib104/show_attend_and_tell_pytorch/blob/main/images/val_2.png "Logo Title Text 1")

#### Test data
##### (1) Generated Caption : A women is sitting at a table with a plate of food.
![](https://github.com/nasib104/show_attend_and_tell_pytorch/blob/main/images/test_1.png "Logo Title Text 1")

##### (2) Generated Caption : A person in a red jacket is standing on a snow covered slope.
![](https://github.com/nasib104/show_attend_and_tell_pytorch/blob/main/images/test_2.png "Logo Title Text 1")
