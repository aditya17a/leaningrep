function accuracy = accuracy(T, P)
% The function to calculate the classification accuracy
% T = Target vector 1xN
% P = Prediction vector 1xN

P(P>=0.5) = 1;
P(P<0.5) = 0;
count = T==P;
accuracy = sum(count)/numel(P);
end