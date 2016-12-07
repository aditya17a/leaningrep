function [W, b, mse]= backProp(Z, Y, W, b, eta)

L = numel(Z);
[E, mse] = evaluateMSE(Y, Z{L});
    for l = L-1:-1:1
        df = Z{l+1}.*(1-Z{l+1});
        dG = df.*E;
        dW = Z{l}*dG';
        db = sum(dG');
        W{l} = W{l}-eta*dW;
        b{l} = b{l}-eta*db;
        E = W{l}*dG;
    end
end