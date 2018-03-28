function [ grad_b, grad_W ] = ComputeGradients( X, Y, P, W, lambda )
%COMPUTEGRADIENTS Compute gradients for W and B (Eq.10 & Eq.11)

    [~, n] = size(X);    
    G = -(Y-P)';
    grad_b = sum(G', 2)/n; %#ok<UDIM>
    grad_W = G'*X'/n + 2*lambda*W;

end

