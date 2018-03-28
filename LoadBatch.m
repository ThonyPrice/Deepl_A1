function [ X, Y, y ] = LoadBatch( fname )
%LOADBATCH: Read data from CIFAR-10 batch file. 
% Return: Image and label data in separate files. 

    A = load(fname);
    X = A.data;
    X = double(X')/255;
    y = double(A.labels') + 1;
    Y = full(ind2vec(y));
    
end

