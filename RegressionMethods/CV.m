
function CVindices = CV(X, folds)

% CVindices  takes data (X) and a positive integer number of folds and
% produces a matrix with number of rows equal to folds, each row containing
% the indices of the data to be used for that fold
    [obs d] = size(X);
    indices = randperm(obs);
    CVindices =  reshape(indices,[folds, obs/folds]);
end
