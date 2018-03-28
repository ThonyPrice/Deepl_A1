function P = EvaluateClassifier( X, W, b )
%EVALUATECLASSIFIER Evaluate network function (Eq.1 & Eq.2)

    [~, N] = size(W*X);
    b = repmat(b,1,N);
    s = W*X + b;
    P = softmax(s);
    
end

