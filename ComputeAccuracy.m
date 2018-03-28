function acc = ComputeAccuracy( X, y, W, b )
%COMPUTEACCURACY: Compute accuracy of network's predictions

    [~, n] = size(X);
    P = EvaluateClassifier(X, W, b);
    [~, predicted] = max(P);
    compare = (predicted - y) == 0;
    correct = sum(compare(:));
    acc = correct / n;
    
end

