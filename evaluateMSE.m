function [E, mse] = evaluateMSE(Y, Z_out)
% Function to calculate the Mean Squared Error at the output layer.
% Outputs:
% E = Error matrix. Size: p x n
% mse = Value of the MSE at the output.
% Inputs:
% Y = Real output (response) data. Size: p x n
% Z_out = Output of the network. Size: p x n

E = Y-Z_out;
mse = mean(dot(E(:),E(:)));


end