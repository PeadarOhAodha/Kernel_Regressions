
function w = dualCost(K, y, dual)
% dualCost computes the mean squared error for kernel ridge regression,
% given a kernel, test data and the learned dual model parameter.
    w = (1/size(K,1)) * (K*dual - y)' * (K*dual - y);
end