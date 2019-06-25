import os
from keras import models
import keras
import sys
from keras.models import load_model

def main():
	weights_path = sys.argv[1]
	json_path = sys.argv[2]
	# weights_path = r'D:\body_model'
	# json_path = r'full_body_classify.json'
	# h5_path = r'full_body_classify.h5'
	with open(os.path.join(weights_path, json_path)) as file:
		_net = models.model_from_json(file.read())
	_net.summary()
	keras.utils.plot_model(_net)

if __name__=='__main__':
	main()
