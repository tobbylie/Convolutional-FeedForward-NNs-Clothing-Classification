import LoadDataModule as loadData
from sklearn.preprocessing import MinMaxScaler
import time
import numpy as np
import matplotlib.pyplot as plt
from keras import optimizers, callbacks
from keras.models import Sequential
from keras.layers import Activation, Conv2D, MaxPooling2D, Dense, Flatten
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

np.random.seed(554433)

# used to get epoch times after each execution
class TimeHistory(callbacks.Callback):
	# list of times
    def on_train_begin(self, logs={}):
        self.times = []

    # get beginning of epoch time
    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    # get end of epoch time
    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

ld = loadData.LoadDataModule()
# load training images and labels from train.zip
images_train, labels_train = ld.load('train')
# load training images and labels from test.zip
images_test, labels_test = ld.load('test')

# Apply min max scaling to data
scaler = MinMaxScaler()
images_train = scaler.fit_transform(images_train)
images_test = scaler.fit_transform(images_test)

# create images train list for reshaping and adding new dimension
images_train_list = []
for indx, image_train in enumerate(images_train):
	image_train = image_train[np.newaxis]
	reshaped_array = image_train.reshape((28, 28))
	images_train_list.append(reshaped_array)

# create images test list for reshaping and adding new dimension
images_test_list = []
for indx, image_test in enumerate(images_test):
	image_test = image_test[np.newaxis]
	reshaped_array = image_test.reshape((28, 28))
	images_test_list.append(reshaped_array)

# convert into numpy arrays
images_train_list = np.array(images_train_list)
images_test_list = np.array(images_test_list)

# create one hot labels for training data
one_hot_labels_train = np.zeros((60000, 10))
for indx, one_hot_label_train in enumerate(one_hot_labels_train):
	one_hot_labels_train[indx][labels_train[indx]] = 1

# create one hot labels for test data
one_hot_labels_test = np.zeros((10000, 10))
for indx, one_hot_label_test in enumerate(one_hot_labels_test):
	one_hot_labels_test[indx][labels_test[indx]] = 1

# reshape lists into dimensions that cnn can accept
images_train_list = images_train_list.reshape(-1, 28, 28, 1)
images_test_list = images_test_list.reshape(-1, 28, 28, 1)

# cnn
model = Sequential()
# convolution layer, 40 feature detectors, kernel size 5x5, relu activation function
# stride 1 and no padding
model.add(Conv2D(40, (5, 5), padding='same', strides=1, activation='relu', input_shape=images_train_list.shape[1:]))
# max pooling layer with pool size 2x2
model.add(MaxPooling2D(pool_size=(2, 2)))
# flatten for fully connected
model.add(Flatten())
# fully connected with 100 neurons with relu activation
model.add(Dense(100, activation='relu'))
# Output layer with 10 neurons and softmax activation
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam',
			loss='categorical_crossentropy',
			metrics=['accuracy'])

# get callback of times
time_callback = TimeHistory()
history = model.fit(images_train_list, one_hot_labels_train, callbacks=[time_callback], epochs=50, batch_size=200)

# loss and accuracy of test data
loss_test, accuracy_test = model.evaluate(images_test_list,one_hot_labels_test, batch_size=200)

# predictions
predictions = model.predict(images_test_list)

# times of epoch completions
times = time_callback.times

# this loop calculates times relative to start
running_total = 0.0
for indx, time in enumerate(times):
	running_total += time
	times[indx] = running_total

# get list of predictions
predictions_list = []
for indx, prediction in enumerate(predictions):
	predictions_list.append(np.argmax(prediction))

# get confusion matrix from test labels and predictions
cm = confusion_matrix(labels_test, predictions_list)
cm = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]

# get accuracy score based on test labels and predictions
acc_score = accuracy_score(labels_test, predictions_list)

# can get class accuracies by getting diagonal of cm
class_accuracies = cm.diagonal()

# get target names for each class
target_names = ['class 0', 'class 1', 'class 2', 'class 3', 'class 4', 'class 5',
				'class 6', 'class 7', 'class 8', 'class 9']

# classification report
cr = classification_report(labels_test, predictions_list, target_names=target_names)

# classification accuracy
print("Classification accuracy: " + str(accuracy_test))
# can also get classification accuracy from acc_score
#print("Classification accuracy: " + str(acc_score))

# confusion matrix
print("Confusion matrix: ")
print(cm)

# get accuracy for each class based on cm diagonal
for indx, class_accuracy in enumerate(class_accuracies):
	print("Class " + str(indx) + " Accuracy: " + str(class_accuracy))

# print classification report
print(cr)

# need to close plt or else image of boot is plotted
plt.close()

# Loss vs Epoch plot
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'])
plt.title('Loss vs. Epoch')
plt.ylabel('Loss')
plt.xlabel('Epoch')

# Loss vs Wall-time plot
plt.subplot(1, 2, 2)
plt.plot(times, history.history['loss'], 'r-')
plt.title('Loss vs. Wall-time')
plt.ylabel('Loss')
plt.xlabel('Wall-time(seconds)')

# so that there are no overlaps in plots
plt.tight_layout()

plt.show()
