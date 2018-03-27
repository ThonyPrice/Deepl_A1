function [ X, Y, y ] = LoadBatch( fname )
%{
...
Function that reads in the data from a CIFAR-10 batch file.
Returns: Image and label data in separate files. 
...
%}

    A = load(fname);
    X = A.data;
    X = double(X')/255;
    y = double(A.labels') + 1;
    Y = full(ind2vec(y));
    
end

