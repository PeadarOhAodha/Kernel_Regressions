
function dual = kRidgeReg(K, y, gamma)
% dual computes the dual model paramter vector for the kernel ridge regression,
% using a kernel matrix of the training data K, the corresponding reponse
% vector y (same length as K) and a penalisation parameter gamma.

   dual = (K + gamma*size(K,1)*eye(size(K,1)))\y; 
end