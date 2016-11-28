function accuracy = accuracy(T, P)
% The function to calculate the classification accuracy
% T = Target vector 1xN
% P = Prediction vector 1xN
count = T==P;
accuracy = 1-sum(count)/numel(P);
end