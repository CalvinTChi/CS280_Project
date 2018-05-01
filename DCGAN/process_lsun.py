import numpy as np
import glob
from scipy.ndimage import imread

def reduce_size(img):
	if img.shape[1] == 256 and img.shape[0] == 256:
		return img
	else:
		wDiff = img.shape[1] - 256
		wMargin = int(wDiff / 2)
		wEnd = wMargin + 256
		hDiff = img.shape[0] - 256
		hMargin = int(hDiff / 2)
		hEnd = hMargin + 256
		img = img[hMargin:hEnd, wMargin:wEnd, :]
		return img

if __name__ == '__main__':
	counter = 0
	images = []
	for filename in glob.glob('./data/bedroom/*.jpg'):
		im = imread(filename)
		im = reduce_size(im)
		images.append(im)
		counter += 1
		if counter % 5000 == 0:
			print(counter)
		if counter == 30000:
			break
	print("saving images...")
	images = np.array(images)
	np.save('./data/bedroom.npy', images)

