function J = ComputeCost( X, Y, W, b, lambda )
%COMPUTECOST Compute cross entropy loss + regularization (Eq.5)

    [~, n] = size(X);
    P = EvaluateClassifier(X, W, b);
    sum1 = sum(-log(diag(Y'*P)));
    W = W.^2;
    J = (1/n) * sum1 + lambda * sum(W(:));

end

