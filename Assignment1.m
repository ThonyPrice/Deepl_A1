%{
...
Function that reads in the data from a CIFAR-10 batch file.
Returns: Image and label data in separate files. 
...
%}

% Global variabes
K = 10;
N = 10000;
d = 3072;

% Load Data
[trainX, trainY, trainy] = LoadBatch('data_batch_1.mat');
[validX, validY, validy] = LoadBatch('data_batch_2.mat');
[testX, testY, testy] = LoadBatch('test_batch.mat');

% Initialize parameters W and b
W = normrnd(0,.01,[K,d]);
b = normrnd(0,.01,[K,1]);

% Check that EvaluateClassifier runs
P = EvaluateClassifier(trainX(:, 1:100), W, b);

