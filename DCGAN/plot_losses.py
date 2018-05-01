import matplotlib.pyplot as plt
import numpy as np 
import getopt
import sys

def main(argv):
	filename = ""
	directory = ""
	title = ""
	try:
		opts, args = getopt.getopt(argv, "d:f:t:")
	except getopt.GetoptError:
		print("Usage: plot_losses.py -d <directory> -f <input_file> -t <title>")
		sys.exit(2)
	for opt, arg in opts:
		if opt == "-d":
			directory = arg
		elif opt == "-f":
			filename = arg
		elif opt == "-t":
			title = arg
	data = np.genfromtxt(directory + filename, delimiter=",")
	data = np.reshape(data, (len(data), 3))
	fig, ax = plt.subplots()
	p1 = plt.plot(data[:, 0], data[:, 1], 'r', label = "discriminator")
	p1 = plt.plot(data[:, 0], data[:, 2], 'b', label = "generator")
	plt.legend(loc='upper right')
	ax.set_xlabel('iterations')
	ax.set_ylabel('loss')
	plt.title(title)
	plt.savefig(directory + "losses.png")

if __name__ == "__main__":
	try:
		arg = sys.argv[1]
	except IndexError:
		print("Usage: plot_losses.py -d <directory> -f <input_file> -t <title>")
		sys.exit(2)
	main(sys.argv[1:])

