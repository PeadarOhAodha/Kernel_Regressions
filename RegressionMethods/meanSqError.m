function mse = meanSqError(w_est, X, y)
% mse computes the mean squared error for a regression model, given the
% leanred model cofficients, matrix of predictors X, and a corresponding
% vector of responses of equal length.
    mse=  1/size(X,1) * (X* w_est - y)' * (X * w_est -y);
    
end