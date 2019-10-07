import LoadDataModule as loadData
from sklearn.preprocessing import MinMaxScaler
import time
import numpy as np
import matplotlib.pyplot as plt
from keras import optimizers, callbacks
from keras.models import Sequential
from keras.layers import Dense, Activation
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

# create one hot labels for training data
one_hot_labels_train = np.zeros((60000, 10))
for indx, one_hot_label_train in enumerate(one_hot_labels_train):
	one_hot_labels_train[indx][labels_train[indx]] = 1

# create one hot labels for test data
one_hot_labels_test = np.zeros((10000, 10))
for indx, one_hot_label_test in enumerate(one_hot_labels_test):
	one_hot_labels_test[indx][labels_test[indx]] = 1

# Apply min max scaling to data
scaler = MinMaxScaler()
images_train = scaler.fit_transform(images_train)
images_test = scaler.fit_transform(images_test)

# Feed forward
model = Sequential()
# 784 neurons with tanh
model.add(Dense(784, activation='tanh', input_dim=784))
# 512 neurons with sigmoid
model.add(Dense(512, activation='sigmoid'))
# 100 neurons with relu
model.add(Dense(100, activation='relu'))
# output with softmax
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam',
			loss='categorical_crossentropy',
			metrics=['accuracy'])

# get callback of times
time_callback = TimeHistory()
history = model.fit(images_train, one_hot_labels_train, callbacks=[time_callback], epochs=50, batch_size=200)

# loss and accuracy of test data
loss_test, accuracy_test = model.evaluate(images_test,one_hot_labels_test, batch_size=200)

# predictions
predictions = model.predict(images_test)

# times of epoch completions
times = time_callback.times

# this for loop calculates times relative to start
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

# get accuracy score based on test labels and predicitions
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
plt.title('Loss vs. Wall-Time')
plt.ylabel('Loss')
plt.xlabel('Wall-Time(seconds)')

# so that there are no overlaps in plots
plt.tight_layout()

plt.show()
