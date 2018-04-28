from shutil import copyfile
import glob

synthetic_cancer = '../cancer_cycle_gan/train_latest/images/*_fake_B.png'
synthetic_healthy = '../cancer_cycle_gan/train_latest/images/*_fake_A.png'
dst_cancer = './synthetic_data/train/cancer/'
dst_healthy = './synthetic_data/train/not/'

for f in glob.glob(synthetic_cancer):
    copyfile(f, dst_cancer + f[40:])
    
for f in glob.glob(synthetic_healthy):
    copyfile(f, dst_healthy + f[40:])

    