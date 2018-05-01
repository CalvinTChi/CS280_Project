# DCGAN
Script to run deep convolutional GAN, taken from [carpedm20](https://github.com/carpedm20/DCGAN-tensorflow/) and edited for this project. The script should be run in 2 stages

+ Training: train GAN, record & output losses, output training images, create checkpoints
+ Image generation: load checkpoints, generate grid images & single images from trained GAN.

# Usage
Train recommended inputs

`
python main.py --dataset [dataset_name] --input_height=[num] --output_height=[num] --data_dir [data_dir] --input_fname_pattern="*.jpg" --trial=[num] --epoch_num=[num] --counter=[num] --output_freq [num] --train 
`

+ `dataset_name`: name of folders created under checkpoint, losses, training_images
+ `data_dir`: directory of files containing image files
+ `trial`: if running different GAN on same dataset, may indicate which trial it is with number. Trial number will be appended to `dataset_name`
+ `epoch_num`: when resuming training, to indicate starting epoch number
+ `counter`: when resuming training, to indicate starting counter number
+ `output_freq`: number of iterations before every training image output

Image generation recommended inputs. Not including `--train` means loading checkpoint as specified by `--dataset` and possibly `--trial` and generating images. 

`
python main.py --dataset [dataset_name] --input_height=[num] --output_height=[num] --batch_size=[num] --trial=[num]
`

+ `batch_size`: number of single images to generate and number of images in a grid 
+ `trial`: specify if trial number was specified during training


