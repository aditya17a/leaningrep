function [model, mse] = mlpStochGD(X, Y, h)
% Multilayer perceptron
% Input:
%   X: d x n data matrix
%   Y: p x n response matrix
%   h: L x 1 vector specify number of hidden nodes in each layer l
% Ouput:
%   model: model structure
%   mse: mean square error

h = [size(X,1);h(:);size(Y,1)]; % Adding input and output layers
N = size(X,2);
W = initializeWeights(h);
b = initializeBiases(h);

maxiter = 3000;
BATCH_SIZE = 500;
lr = 0.1;
eta = lr*1/BATCH_SIZE;


mse = zeros(1,maxiter);
for iter = 1:maxiter
%     timer start
    if mod(iter,100) == 1
    tic()
    end
    mse_mini = [];
    for ll = 1:BATCH_SIZE: N
      if ll+BATCH_SIZE-1<N
        X_mini = X(:,ll:ll+BATCH_SIZE-1);
        Y_mini = Y(:,ll:ll+BATCH_SIZE-1);
      else
        X_mini = X(:,ll:end);
        Y_mini = Y(:,ll:end);
      end
  %     forward
      Z = forwardPass(X_mini, W, b);
  %     backward    
      [W, b, mse_batch] = backProp(Z, Y_mini, W, b, eta);
      mse_mini = [mse_mini, mse_batch];
    end
    mse(iter) = mean(mse_mini)/BATCH_SIZE;
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