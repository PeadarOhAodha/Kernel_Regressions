function K = gaussKernel(X, sigma)

% ---- REFERENCE: Code for how to vectorise this gamma kernel function for computational 
%efficiency (instead of looping through each data pair)
% was referenced from http://www0.cs.ucl.ac.uk/staff/M.Herbster/boston/rbf.m

temp=X*X'/sigma^2; %Computing matrix norm. Utilising symmetry of kernel matrix.
temp=temp-ones(size(X,1),1)*diag(temp)'/2; %Correcting diagonal
K=exp(temp-diag(temp)*ones(1,size(X,1))/2);
