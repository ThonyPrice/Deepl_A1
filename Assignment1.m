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
lambda = 0;
GDparams = GDparams(100, .01, 20);

% Testing popose only
rng(400);

% Load Data
[trainX, trainY, trainy] = LoadBatch('data_batch_1.mat');
[validX, validY, validy] = LoadBatch('data_batch_2.mat');
[testX, testY, testy] = LoadBatch('test_batch.mat');

% Initialize parameters W and b
W = normrnd(0,.01,[K,d]);
b = normrnd(0,.01,[K,1]);

% Check that EvaluateClassifier runs
% P = EvaluateClassifier(trainX(:, 1:100), W, b);

% Test ComputeCost
% J = ComputeCost(trainX(:, 1:100), trainY(:, 1:100), W, b, 0.1);

% Compare numericallly and analytically computed gradients
P = EvaluateClassifier(trainX(:, 1), W, b);
[agrad_b, agrad_W] = ComputeGradients(trainX(:, 1), trainY(:, 1), P, W, lambda);
[ngrad_b, ngrad_W] = ComputeGradsNumSlow(trainX(:, 1), trainY(:, 1), W, b, lambda, 1e-6);

% Run!
epochs = (1:GDparams.n_epochs);
cost = zeros(1,GDparams.n_epochs);
accuracy = zeros(1,GDparams.n_epochs);
for epoch = epochs
    [W, b] = MiniBatchGD(trainX, trainY, GDparams, W, b, lambda);
    cost(epoch) = ComputeCost(trainX, trainY, W, b, lambda);
    accuracy(epoch) = ComputeAccuracy(trainX, trainy, W, b);
    msg = ['Epoch: ', num2str(epoch)];
    msg = [msg [' Cost: ', num2str(cost(epoch))]];
    msg = [msg [' Accuracy: ', num2str(accuracy(epoch))]];
    disp(msg);
end


disp('EOF');


