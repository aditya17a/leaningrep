function y = sigmoid(x)
% Sigmod function
y = exp(-log1pexp(-x));