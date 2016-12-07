function b = initializeBiases(layers)
% Function to initialize weights randomly.
% Outputs:
% W = Initialized weight matrix cell. Size depends on number of layers.
% Inputs:
% layers = Total number of layers in the network (including input, hidden and output layers).
L = numel(layers);

for l = 1:L-1
    b{l} = randn(1,layers(l+1));
end
end