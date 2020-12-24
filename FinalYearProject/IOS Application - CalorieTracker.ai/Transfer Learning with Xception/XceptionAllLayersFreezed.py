#Title: Custom Deep Convolutional Neural Network, Xecption with Transfer Learning
#Author: Mohammad Ashraful Islam Sadi
#Date: 20/04/2019
#Code version: Python 3.7
#The code is described quite well in Section 5.1: Implementation
#It was originally inspired from Keras documentation but it has been changed a lot now to fit the needs of the project.
#Keras documentation: https://keras.io/

#First all the necessary libraries from Python are imported.
import keras
from keras.layers import Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.applications.xception import Xception
from keras import models
from keras.models import Sequential
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import glob as glob

#The necessary variables for directories, algorithms and  are defined in this section.
image_format = (299, 299)
inputShape = (299, 299, 3)
trainingDirectory = 'Training'
validationDirectory = 'Testing'
imageNetWeights = 'imagenet'
reluAlgorithm = 'relu'
softmaxAlgorithm = 'softmax'
categoricalCrossentropyAlgorithm = 'categorical_crossentropy'
adamOptimizer = Adam(lr=0.0001)
trainingBatchSize = 100
testingBatchSize = 10
epochNumber = 30
epochSteps = 10
classificationClasses = 8
colorMode = 'rgb'
validationAccuracy = 'val_acc'
validationLoss = 'val_loss'
trainingAccuracy = 'acc'
trainingLoss = 'loss'


#The pre trained algorithm, Xception is initialised with Imagenet weights and without the last layer.
preTrainedAlgorithm = Xception(input_shape = inputShape, weights = imageNetWeights,
                   include_top = False)

#All the layers in the pre-trained algorithm is kept frozen to keep the weight values constant.
#Changing this parameter will significantly change the results, which is shown in the Appendix section.
for layer in preTrainedAlgorithm.layers:
   layer.trainable = False

#A sequential model (stack of linear layers) is defined. The pre-trained algorithm is added to the sequential model and the 4D tensor output of the model is flattened.
#Then two dense layers with 512 neurons each are added to it which acts as fully connected layers, with Dropout = 0.5 between them to avoid overfitting.
#The activation algorithm used here to decrease linearity is ReLU.
#The final dense layer added is a logistic regression classifier which consists of eight neurons as parameters as there are eight classification types.
#The activation algorithm used here is Softmax.
customModel = Sequential()
customModel.add(preTrainedAlgorithm)
customModel.add(layers.Flatten())
customModel.add(layers.Dense(512, activation = reluAlgorithm))
customModel.add(layers.Dropout(0.5))
customModel.add(layers.Dense(512, activation = reluAlgorithm))
customModel.add(layers.Dense(classificationClasses, activation = softmaxAlgorithm))
customModel.summary()

#This is a special mix of two algorithms in Keras called ModelCheckpoint and EarlyStopping. Early Stopping monitors the validation acccuracy value and
#stops the training process at the next epoch if the value decreases from the previous epoch. ModelCheckpoint monitors and saves the model with the least
#validation loss value in .h5 format.
callbacks_list = [
                  keras.callbacks.ModelCheckpoint(
                                                  filepath='best_model.{epoch:02d}-{val_loss:.2f}.h5',
                                                  monitor= validationLoss, save_best_only=True, mode= min, period= 0),
                  keras.callbacks.EarlyStopping(monitor= validationAccuracy, patience= 0, mode= max)
                  ]

#The custom model is compiled with loss algorithm, Categorical_Crossentropy, Adam optimizer and accuracy metrics.
customModel.compile(loss = categoricalCrossentropyAlgorithm,
              optimizer = adamOptimizer,
              metrics = ['accuracy'])


#ImageDataGenerator helps augmenting the training data to increase variations in the dataset.
trainingDataGenerator = ImageDataGenerator(rescale = 1./299, shear_range = 0.2,
                                           zoom_range = 0.2,
                                           horizontal_flip = True,
                                           rotation_range = 20)

testingDataGenerator = ImageDataGenerator(rescale = 1./299)

#Final step of preparation for the training images
augmentedTrainingData = trainingDataGenerator.flow_from_directory(trainingDirectory,
                                                 target_size = image_format,
                                                 batch_size = trainingBatchSize,
                                                 class_mode = 'categorical',
                                                 color_mode= colorMode,
                                                 shuffle=True)

#Final step of preparation for the validation data set.
augmentedTestingData = testingDataGenerator.flow_from_directory(validationDirectory,
                                            target_size = image_format,
                                            batch_size = testingBatchSize,
                                            class_mode = 'categorical',
                                            color_mode = colorMode,
                                            shuffle = False)

#The function below starts the training process with the necessary parameters
print('Training Started')
learnedModel = customModel.fit_generator(augmentedTrainingData,
                              validation_data = augmentedTestingData,
                              epochs = epochNumber,
                              steps_per_epoch = epochSteps,
                              validation_steps = len(augmentedTestingData),
                              callbacks= callbacks_list, verbose =1)
print('Training Ended')

#The below code is taken from the matplotlib library to generate graphs based on the training history,
#which includes training and validation accuracy and loss values.
#accuracies
plt.plot(learnedModel.history['acc'], label='train acc')
plt.plot(learnedModel.history['val_acc'], label='val acc')
plt.legend()
plt.show()

# loss
plt.plot(learnedModel.history['loss'], label='train loss')
plt.plot(learnedModel.history['val_loss'], label='val loss')
plt.legend()
plt.show()

#Saving the trained model after observing the graphs generated above. If the accuracy metrics are good then the model is saved.
#Otherwise the training process is interrupted if the accuracy and loss graphs demonstrate high overfitting.
import tensorflow as tf
from keras.models import load_model
customModel.save('XceptionAllLayersFreezed2.h5')


#This whole block of code until the end is taken from a course in Udemy.
#Course Link: https://www.udemy.com/advanced-computer-vision/learn/v4/t/lecture/9339972?start=120
#The objective of the code block was to generate confusion matrix and classification reportwhich will be used in Evaluation.

Y_pred = customModel.predict_generator(augmentedTestingData, len(augmentedTestingData))
y_pred = np.argmax(Y_pred, axis = 1)
print('Confusion Matrix')
print(confusion_matrix(augmentedTestingData.classes, y_pred))
print('Classification Report')
classificationClasses = ['KFC Chicken(Thigh) 285 kcal',
                         'KFC Fillet Rice Box 448 kcal',
                         'KFC Fillet Tower Burger 650 kcal',
                         'KFC Fries 250 kcal',
                         'KFC Popcorn Chicken 285 kcal',
                         'KFC Zinger Burger 450 kcal',
                         'KFC Zinger Stacker Burger 780 kcal',
                         'Mcdonalds Big Mac Burger 508 kcal']

print(classification_report(augmentedTestingData.classes, y_pred,
                            target_names = classificationClasses))


image_files = glob(trainingDirectory + '/*/*.jp*g')
valid_image_files = glob(validationDirectory + '/*/*.jp*g')

# useful for getting number of classes
folders = glob(trainingDirectory + '/*')

# get label mapping for confusion matrix plot later
#test_gen = gen.flow_from_directory(valid_path, target_size=IMAGE_SIZE)
#print(test_gen.class_indices)
print(augmentedTestingData.class_indices)
labels = [None] * len(augmentedTestingData.class_indices)
for k, v in augmentedTestingData.class_indices.items():
  labels[v] = k

def get_confusion_matrix(data_path, N):
  # we need to see the data in the same order
  # for both predictions and targets
  print("Generating confusion matrix", N)
  predictions = []
  targets = []
  i = 0
  for x, y in testingDataGenerator.flow_from_directory(data_path, target_size=image_format, shuffle=False, batch_size=testingBatchSize * 2):
    i += 1
    if i % 50 == 0:
      print(i)
    p = customModel.predict(x)
    p = np.argmax(p, axis=1)
    y = np.argmax(y, axis=1)
    predictions = np.concatenate((predictions, p))
    targets = np.concatenate((targets, y))
    if len(targets) >= N:
      break

  cm = confusion_matrix(targets, predictions)
  return cm


cm = get_confusion_matrix(trainingDirectory, len(image_files))
print(cm)
valid_cm = get_confusion_matrix(validationDirectory, len(valid_image_files))
print(valid_cm)


from util import plot_confusion_matrix
plot_confusion_matrix(cm, labels, title='Train confusion matrix')
plot_confusion_matrix(valid_cm, labels, title='Validation confusion matrix')
