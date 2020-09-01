﻿/*
This is will train a small Convolutional Neural Network on the MNIST dataset. It demonstrates
how to use image data with GNN (see  mnist_as_real.ecl on how to properly prepare the data for a tensor)
as well as demnostrates how to use the GNN that is GPU enabled.


Trains a 34k parameter CNN on 60k MNIST images and tests on an additional, unseen, 8k images for
measuring accuracy. Trains for 10 epochs and a mini-batch size of 128.


Expected Accuracy:  You can expect over 96% accuracy accross 2 GPUs acrros 2 physcal computer
Expect Training Time: You can expect the whole Workunit to complete in under 6 minutes using 2 
											physcal computers, 2 NVIDIA K80 GPUs, and a 2 Thor cluster.

*/

#option('outputLimit',2000);
#option('hthorMemoryLimit',10000);
IMPORT Python3 AS Python;
IMPORT $.^.^ AS GNN;
IMPORT GNN.Tensor;
IMPORT GNN.Internal.Types AS iTypes;
IMPORT GNN.Types;
IMPORT GNN.GNNI;
IMPORT GNN.Utils;
IMPORT GNN.Internal AS Int;
IMPORT Std.System.Thorlib;



numOfGPUs := 2; //16 for total system GPU count, just used for workunit naming
aggregateInterval := 1; //how many times do you want the weights to be aggregated per epoch
numOfPhysicalMachines := 2; //-1 makes TF to use CPU across the board, set the number of physical computer, assumes number of GPUs are equal accross machines and assumes number of Thor nodes is equal to total number of GPUs in system


IMPORT mnist_as_real_big as mnist;
trainCount := 60000; //This is all of them
testCount := 8196; //Most of the test, fits easier accross multiple GPUs
numEpochs := 10;

#WORKUNIT('name', 'MNIST Test with '+numOfGPUs+' GPUs - TrainCount: '+trainCount + ' - aggs per epoch: ' + aggregateInterval + ' - epochs:' + numEpochs); 

miniBatch := 128; //32 is Stock-GNN
oldBatchSize := 128;

featureCount := 784;
weightAggregateInterval := trainCount / (numOfGPUs * aggregateInterval); //old "batchSize"


trainData := CHOOSEN(mnist.train,trainCount);
testData := CHOOSEN(mnist.test,testCount);

myXTensData := NORMALIZE(trainData, featureCount,
                     TRANSFORM(Tensor.R4.TensData,
                       SELF.indexes := [LEFT.id, (COUNTER-1) DIV 28 + 1, (COUNTER-1) % 28 +1, 1],
                       SELF.value := LEFT.pixel[COUNTER]));

myXTensDataTest := NORMALIZE(testData, featureCount,
                     TRANSFORM(Tensor.R4.TensData,
                       SELF.indexes := [LEFT.id, (COUNTER-1) DIV 28 + 1, (COUNTER-1) % 28 +1, 1],
                       SELF.value := LEFT.pixel[COUNTER]));											 
//OUTPUT(COUNT(myXTensData));											 
											 
// Each source record becomes 10 Y (in this case "label" tensor cells (one per class value)
// using One-Hot encoding.
// But only the record associated with the class (i.e. value of Y)
// will be 1.  The others will be zero.  Since the TensData format
// is sparse, we just skip the zero cells.
myYTensData := NORMALIZE(trainData, 10,
                     TRANSFORM(Tensor.R4.TensData,
                       SELF.indexes := [LEFT.id, COUNTER],
                       SELF.value := IF(LEFT.label != COUNTER - 1, SKIP, 1)));
											 
myYTensDataTest := NORMALIZE(testData, 10,
                     TRANSFORM(Tensor.R4.TensData,
                       SELF.indexes := [LEFT.id, COUNTER],
                       SELF.value := IF(LEFT.label != COUNTER - 1, SKIP, 1)));											 
											 
// Now we convert the Tensor Data to a Tensor dataset by calling MakeTensor()
myXTensor := Tensor.R4.MakeTensor([0,28,28,1], myXTensData);
myYTensor := Tensor.R4.MakeTensor([0,10], myYTensData);

myXTensorTest := Tensor.R4.MakeTensor([0,28,28,1], myXTensDataTest);
myYTensorTest := Tensor.R4.MakeTensor([0,10], myYTensDataTest);


instancesPerSlice := COUNT(myXTensor[1].densedata)/featureCount;
//OUTPUT(myXTensor, NAMED('SeeingTheSlices')); 
//OUTPUT(instancesPerSlice, NAMED('InstancesPerSlice'));

//OUTPUT(myXTensor, NAMED('x1'));
//OUTPUT(myYTensor, NAMED('y1'));

ldef := [	'''layers.Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(28,28,1))''',
					'''layers.MaxPooling2D(pool_size=(2, 2))''',
					'''layers.Conv2D(64, kernel_size=(3, 3), activation="relu")''',
					'''layers.MaxPooling2D(pool_size=(2, 2))''',
					'''layers.Flatten()''',
					'''layers.Dropout(0.5)''',
					'''layers.Dense(10, activation="softmax")''']; // 34,826 trainable parameters

					
compileDef := '''compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.SGD(learning_rate=0.0005, momentum=0.1), metrics=["accuracy"])''';


// GetSession must be called before any other functions
s := GNNI.GetSession(numOfPhysicalMachines);
// DefineModel is dependent on the Session
//   ldef contains the Python definition for each Keras layer
//   compileDef contains the Keras compile statement.

mod := GNNI.DefineModel(s, ldef, compileDef);

// GetWeights returns the initialized weights that have been synchronized across all nodes.
wts := GNNI.GetWeights(mod);
OUTPUT(wts, NAMED('InitWeights'));

mod2 := GNNI.Fit(mod, myXTensor, myYTensor, batchSize := weightAggregateInterval, miniBatch := miniBatch, numEpochs := numEpochs);

//OUTPUT(mod2, NAMED('mod2_FIT'));

// GetLoss returns the average loss for the final training epoch
losses := GNNI.GetLoss(mod2);
OUTPUT(losses, NAMED('Losses'));

// EvaluateMod computes the loss, as well as any other metrics that were defined in the Keras
// compile line.
metrics := GNNI.EvaluateMod(mod2, myXTensorTest, myYTensorTest);

OUTPUT(metrics, NAMED('metrics'));











