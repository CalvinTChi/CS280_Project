from shutil import copyfile
import glob

synthetic_cancer = '../testing/test_latest/images/*_fake_B.png'
synthetic_healthy = '../testing/test_latest/images/*_fake_A.png'
dst_cancer = './synthetic_validation/cancer/'
dst_healthy = './synthetic_validation/not/'



for f in glob.glob(synthetic_cancer):
    copyfile(f, dst_cancer + f[30:])
    
for f in glob.glob(synthetic_healthy):
    copyfile(f, dst_healthy + f[30:])

    