
function w = RidgeR(X, y, r)
% RidgeR computes the ridge regression coefficient vector, given a
% training matrix X, and an equal length response vector y and penalisation
% parameter r.

    w = ((X'*X) + size(X,1)*r*eye(size(X,2)))\X'*y;
end