function [ Wstar, bstar ] = MiniBatchGD( X, Y, GDparams, W, b, lambda )
%MINIBATCHGD Summary of this function goes here
%   Detailed explanation goes here
    
    [~, N] = size(X);
    n_batch = GDparams.n_batch;

    % Mk batches
    for j=1:N/n_batch
        j_start = (j-1)*n_batch + 1;
        j_end = j*n_batch;
        Xbatch = X(:, j_start:j_end); 
        Ybatch = Y(:, j_start:j_end);
        % Update equations
        P = EvaluateClassifier(Xbatch, W, b);
        [grad_b, grad_W] = ComputeGradients(Xbatch, Ybatch, P, W, lambda);
        Wstar = W - GDparams.eta * grad_W;
        bstar = b - GDparams.eta * grad_b;
        W = Wstar;
        b = bstar;
    end
end

