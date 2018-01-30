from keras.layers import Dense, Flatten, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.layers.wrappers import TimeDistributed
from keras.layers.convolutional import Conv2D, MaxPooling2D

def lstm(self):
	"""Build a simple LSTM network. We pass the extracted features from
	our CNN to this model predomenently."""
	# Model.
	model = Sequential()
	model.add(LSTM(2048, return_sequences=False,
				   input_shape=self.input_shape,
				   dropout=0.5))
	model.add(Dense(512, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(self.nb_classes, activation='softmax'))

	return model
