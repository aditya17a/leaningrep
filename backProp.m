function [W, mse]= backProp(Z, Y, W, eta)

L = numel(Z);
[E, mse] = evaluateMSE(Y, Z{L});
    for l = L-1:-1:1
        df = Z{l+1}.*(1-Z{l+1});
        dG = df.*E;
        dW = Z{l}*dG';
        W{l} = W{l}+eta*dW;
        E = W{l}*dG;
    end
end