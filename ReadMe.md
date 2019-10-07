# Classifying Clothes

	This program utilizes a feed-forward neural network and two convolutional neural
	networks in order to classify clothes from compressed data files. 

## What's included:
```
	LoadDataModule.py - .py file utilized in order to uncompress and load data
	from compressed data files.

	task_1.2_metrics.png - metrics screenshot for task_1.2

	task_1.2.plot.png - plot for task_1.2

	task_1.2.py - Python code for task_1.2

	task_2.2_metrics.png - metrics screenshot for task_2.2

	task_2.2.plot.png - plot for task_2.2

	task_2.2.py - Python code for task_2.2

	task_3.2_metrics.png - metrics screenshot for task_3.2

	task_3.2.plot.png - plot for task_3.2

	task_3.2.py - Python code for task_3.2

	train_images.zip - 60,000 samples of 28x28 grayscale images. The data is of size
	60000 x 784. Each pixel has a single intensity value which is an integer between
	0 and 255.

	train_labels.zip - 60,000 samples of 10 classes for the images in the given train_images. Class details are mentioned below.

	test_images.zip - 10,000 samples of 28 x 28 grayscale image. The data is of size 10000 x 784. Each pixel has a single intensity value which is an integer between 0 and 255.

	test_labels.zip - 10,000 samples from 10 classes for the images in the given test_images. Class details are listed below.
```
# Class labels:

| Class | Type of Dress |
| --- | --- |
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

## Model Descriptions:

### 1.2 - Feed Forward neural network
		- Hidden layer 1: 784 neurons with hyperbolic tangent activation function in each neuron.
		- Hidden layer 2: 512 neurons, with sigmoid activation function in each of the neuron.
		- Hidden layer 3: 100 neurons, with rectified linear activation function in each of the neuron.
		- Output layer: 10 neurons representing the 10 classes, and obviously softmax activation function for each of the 10 neurons.
		- Loss function: Categorical cross-entropy
		- Epochs: 50
		- Batch Size: 200

### 2.2 - Small Convolutional neural network
		- Convolution layer having 40 feature detectors, with kernel size 5 x 5, and rectifier as the activation function, with stride 1 and no-padding.
		- A max-pooling layer with pool size 2x2.
		- Fully connected layer with 100 neurons, and rectifier as the activation function.
		- Output layer containing 10 neurons, softmax as activation function.
		- Loss function: Categorical cross-entropy
		- Epochs: 50
		- Batch Size: 200

### 3.2 - Bigger Convolutional neural network
		- Convolution layer having 48 feature detectors, with kernel size 3 x 3, and rectifier as the activation function, with stride 1 and no padding.
		- A max-pooling layer with pool size 2x2.
		- Convolution layer having 96 feature detectors, with kernel size 3 x 3, and rectifier as the activation function, with stride 1 and no padding.
		- A max-pooling layer with pool size 2x2.
		- Fully connected layer with 100 neurons, and rectifier as the activation function.
		- Output layer containing 10 neurons, softmax as activation function.
		- Loss function: Categorical cross-entropy
		- Epochs: 50
		- Batch Size: 200

## Use:
	Once in directory with contents, run python3 task_1.2.py, task_2.2.py or
	task_3.2.py in order to train a model, evaluate as well as predict on a
	test dataset. Ultimately metrics will be produced describing: classification
	accuracy, confusion matrix, class-wise accuracy, precision, recall and f-1
	score. Two plots will also be produced showing Loss vs. Epoch and Loss vs.
	Wall-Time.
