function P = EvaluateClassifier( X, W, b )
%EVALUATECLASSIFIER Evaluate network function

    %size(X)
    %size(W)
    %size(b)
    %size(W*X)

    % Equation 1
    tmp = W*X;
    [~, N] = size(tmp);
    b = repmat(b,1,N);
    s = tmp + b;

    % Equation 2
    P = softmax(s);
    
end

