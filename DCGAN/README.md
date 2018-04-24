# DCGAN

Script to run deep convolutional GAN for lymphoma cancer classification

# Setup

1. Add `X.npy` and `Y.npy` to `./data/lymph` folder.
2. Uncompress files in `./checkpoint/lymph_cancer_64_50_50` 
3. Create folders `./data/lymph`, `./lymph_cancer_images`, `./lymph_normal_images`

# Execution

To run DCGAN for lymphoma cancer images

`python main.py --dataset lymph_cancer --input_height=50 --output_height=50  --train`

To run DCGAN for lymphoma normal images

`python main.py --dataset lymph_normal --input_height=50 --output_height=50  --train`

After `n` number of iterations, DCGAN will (1) output a batch of generated images combined as a grid in `samples` (2) output batch of generated images individually under `lymph_cancer_images` or `lymph_normal_images`, along with the corresponding `npy` file. 

For output files, only files in `lymph_cancer_images` or `lymph_normal_images` will be overwritten for each iteration.

