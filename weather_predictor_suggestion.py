import tensorflow as tf
import pandas as pd
import numpy as np
from random import shuffle
from matplotlib import pyplot as plt

np.set_printoptions(precision=3)

# Read csv data into pandas dataframe
df = pd.read_csv("weatherData/weatherTable.csv")

# Convert dataframe into numpy array
data = np.array(df.values)

# Extract real columns
realData = data[:, [2, 3, 4, 5, 7, 8]]

numberOfColumns = realData.shape[1]
learningRate = 0.01
numberOfEpochs = 50000

# How much data should be used for training
trainingSetRatio = 0.8

# How many samples in sequence
sampleSequence = 5

# How many samples away from end of sample sequence should we predict. 0 is next prediction of next sample
predictionTime = 0

trainingData = np.empty(shape=(0, sampleSequence*numberOfColumns))
validationData = np.empty(shape=(0, sampleSequence*numberOfColumns))
trainingTarget = np.empty(shape=(0, numberOfColumns))
validationTarget= np.empty(shape=(0, numberOfColumns))

# Create a randomly shuffled selection of data
availableShuffledIndices = [*range(realData.shape[0] - sampleSequence - predictionTime)]
shuffle(availableShuffledIndices)

# Split data into training and validation data sets
for i in range(len(availableShuffledIndices)):

    currentIndex = availableShuffledIndices[i]

    # Extract sample
    s = [x + currentIndex for x in range(sampleSequence)]
    st = currentIndex + sampleSequence + predictionTime
    sample = np.expand_dims(np.reshape(realData[s, :], (sampleSequence*numberOfColumns)), 0)
    sampleTarget = np.expand_dims(realData[st, :], 0)

    # Create training data set
    if i < len(availableShuffledIndices) * trainingSetRatio:
        trainingData = np.append(trainingData, sample, 0)
        trainingTarget = np.append(trainingTarget, sampleTarget, 0)

    # Create validation data set
    else:
        validationData = np.append(validationData, sample, 0)
        validationTarget = np.append(validationTarget, sampleTarget, 0)

# Create calculation tree
input = tf.placeholder(dtype=tf.float32, shape=(None, trainingData.shape[1]), name="input")
target = tf.placeholder(dtype=tf.float32, shape=(None, trainingTarget.shape[1]), name="target")

normInput = tf.nn.batch_normalization(x=input, mean=0, variance=1, offset=0, scale=1, variance_epsilon=1e-7)
normTarget = tf.nn.batch_normalization(x=target, mean=0, variance=1, offset=0, scale=1, variance_epsilon=1e-7)

layer1 = tf.layers.dense(inputs=normInput,
                         units=trainingData.shape[1]*8,
                         activation=None,
                         kernel_initializer=tf.initializers.random_normal,
                         bias_initializer=tf.initializers.random_normal,
                         name="layer1")

layer1Activation = tf.nn.leaky_relu(features=layer1, alpha=0.1)

layer2 = tf.layers.dense(inputs=layer1Activation,
                         units=trainingData.shape[1]*4,
                         activation=None,
                         kernel_initializer=tf.initializers.random_normal,
                         bias_initializer=tf.initializers.random_normal,
                         name="layer2")

layer2Activation = tf.nn.leaky_relu(features=layer2, alpha=0.1)
#layer2Activation = tf.nn.sigmoid(layer2)

layer3 = tf.layers.dense(inputs=layer2Activation,
                         units=trainingTarget.shape[1],
                         activation=None,
                         kernel_initializer=tf.initializers.random_normal,
                         bias_initializer=tf.initializers.random_normal,
                         name="layer3")

# Remove batch dimension if size 0
prediction = tf.squeeze(layer3, name="squeeze")

loss = tf.reduce_mean(tf.squared_difference(prediction, target, name="loss"))

optimizer = tf.train.AdamOptimizer(learning_rate=learningRate)

train_op = optimizer.minimize(loss)

trainingLossLog = []
validationLossLog = []

fig1 = plt.figure(figsize=(10, 5))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(trainingData.shape)
    for i in range(numberOfEpochs):
        _, trainingLoss = sess.run([train_op, loss],
                                   feed_dict={input : trainingData[:,:(numberOfColumns*sampleSequence)],
                                              target : trainingData[:,-(numberOfColumns):]})
        pred, validationLoss = sess.run([prediction, loss], feed_dict={input : validationData[:,:(numberOfColumns*(sampleSequence))],
                                              target : validationData[:,-(numberOfColumns):]})

        print("\rPrediction: {} Target {} Training loss: {:.2f} Validation loss: {:.2f}".format(pred[0,:], validationData[0,-(numberOfColumns):], trainingLoss, validationLoss), end="")

        trainingLossLog.append(np.log(trainingLoss))
        validationLossLog.append(np.log(validationLoss))

        if i % 1000 == 0:
            plt.plot(trainingLossLog, 'r')
            plt.plot(validationLossLog, 'b')
            plt.xlabel("Episode")
            plt.ylabel("Loss")
            plt.title("Training and validiation log loss")
            plt.pause(0.05)

plt.show()

