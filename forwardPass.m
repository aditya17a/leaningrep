function Z = forwardPass(X, W)
% Forward pass for the neural network.
% Inputs:
% W = Initialized weight matrix cell. Size depends on number of layers.
% X = Input matrix d x n
%
% Outputs:
% Z = Activation cell. Contains activations for all layers of the network. First
% matrix in the cell is the input matrix, last matrix is the output matrix.

L = numel(W) + 1;
%Z = cell(L,1);
Z{1} = X;
for l = 2:L
       Z{l} = sigmoid(W{l-1}'*Z{l-1});
end
end
