function [model, mse] = mlp(X, Y, h)
% Multilayer perceptron
% Input:
%   X: d x n data matrix
%   Y: p x n response matrix
%   h: L x 1 vector specify number of hidden nodes in each layer l
% Ouput:
%   model: model structure
%   mse: mean square error

h = [size(X,1);h(:);size(Y,1)]; % Adding input and output layers

W = initializeWeights(h);
b = initializeBiases(h);
#Z = cell(L);
#Z{1} = X;
lr = 10;
eta = lr*1/size(X,2);
%eta = 0.01;
maxiter = 7000;
mse = zeros(1,maxiter);
for iter = 1:maxiter
%     timer start
    if mod(iter,100) == 1
    tic()
    end
%     forward
    Z = forwardPass(X, W, b);
%     backward    
    [W, b, mse(iter)] = backProp(Z, Y, W, b, eta);
    
%     print results
    if mod(iter,100) == 0
        disp(['Iteration: ', num2str(iter), '|Error: ', num2str(mse(iter))])
        toc()
        fflush(stdout);
    end
end
mse = mse(1:iter);
model.W = W;
model.b = b;