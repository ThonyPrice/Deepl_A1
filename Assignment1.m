% Global variabes
K = 10;
N = 10000;
d = 3072;
lambda = 0;
GDparams = GDparams(100, .01, 40);
rng(400);

% Load Data
[trainX, trainY, trainy] = LoadBatch('data_batch_1.mat');
[validX, validY, validy] = LoadBatch('data_batch_2.mat');
[testX, testY, testy] = LoadBatch('test_batch.mat');

% Initialize parameters W and b
W = normrnd(0,.01,[K,d]);
b = normrnd(0,.01,[K,1]);

% Compare numericallly and analytically computed gradients
%P = EvaluateClassifier(trainX(:, 1), W, b);
%[agrad_b, agrad_W] = ComputeGradients(trainX(:, 1), trainY(:, 1), P, W, lambda);
%[ngrad_b, ngrad_W] = ComputeGradsNumSlow(trainX(:, 1), trainY(:, 1), W, b, lambda, 1e-6);

% Train
epochs = (1:GDparams.n_epochs);
train_cost = zeros(1,GDparams.n_epochs);
valid_cost = zeros(1,GDparams.n_epochs);
% Record initial values
epochs = [0 epochs];
train_cost = [ComputeCost(trainX, trainY, W, b, lambda) train_cost];
valid_cost = [ComputeCost(validX, validY, W, b, lambda) valid_cost];
accuracy = zeros(1,GDparams.n_epochs);
% Update equations
for epoch = 2:GDparams.n_epochs+1;
    [W, b] = MiniBatchGD(trainX, trainY, GDparams, W, b, lambda);
    train_cost(epoch) = ComputeCost(trainX, trainY, W, b, lambda);
    valid_cost(epoch) = ComputeCost(validX, validY, W, b, lambda);
    accuracy(epoch) = ComputeAccuracy(trainX, trainy, W, b);
    msg = ['Epoch: ', num2str(epoch-1)];
    msg = [msg [' tCost: ', num2str(train_cost(epoch))]];
    msg = [msg [' vCost: ', num2str(valid_cost(epoch))]];
    msg = [msg [' Accuracy: ', num2str(accuracy(epoch))]];
    disp(msg);
end

% Plot loss functions
close all
plot(epochs, train_cost, 'g', epochs, valid_cost, 'r');
ylabel('loss'); xlabel('epoch');
legend('training loss','validation loss');

% Test Accuracy
disp(['Test Acc: ', num2str(ComputeAccuracy(testX, testy, W, b))]);

% Visualize weight matrix
for i=1:10
    im = reshape(W(i, :), 32, 32, 3);
    s_im{i} = (im - min(im(:))) / (max(im(:)) - min(im(:)));
    s_im{i} = permute(s_im{i}, [2, 1, 3]);
end
run(montage(s_im(:, :, :, 1:1), 'Size', [1,10]));

disp('EOF');
