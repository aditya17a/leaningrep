clear; close all;
load ('figs.mat');
train_x = Diamod.train_x;
train_y = Diamod.train_y;
test_x = Diamod.test_x;
test_y = Diamod.test_y;
h = [4];

[model,mse] = mlp(train_x',train_y',h);
plot(mse);

predictions = mlpPred(model,test_x');
acc = accuracy(test_y', predictions);

disp(['Test Accuracy: ' num2str(acc)]);

