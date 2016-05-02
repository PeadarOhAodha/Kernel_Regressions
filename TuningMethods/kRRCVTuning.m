
function [s , g, CVmse, Sigma, Gamma] = kRRCVTuning(X_train,y_train, numFolds, param1, param2)
% kRRCVTuning  exectues and tunes kernel ridge regression routine, that uses a gaussian kernel with
% respect to two mandatory hyper parmaters:
%    param1:  sigma hyper parameter for gaussian kernel
%    param1:  gamma regularisation parameter for dual form ridge regression
%
% Exepcts the additional mandatory inputs:
%   X_train = matrix of predictors
%   Y_train = corresponding vector of response (must be same length as X_train columns)
%   numFolds =  number of folds to use with cross validation

% Produces outputs:
%   s = running values of sigma to compare to cross validation error as the
%   routine progresses
%   g = running values of gamma to compare to cross validation error as the
%   routine progresses
%   CVMSE = array of mean over CV folds of cross validation errors for each
%   hyper parameter combination
%   Sigma = best sigma chosen
%   Gamma = best gamma chosen
    

    %call method for giving matrix of randomly generated data indices for 
    %each cross validation fold - see CoreMethods for more details
    CVin = CV(X_train, numFolds); 
    trainSize = size(X_train, 1);
    
    %generate array of hyper parameter pairs for efficiency
    params = {param1, param2};
    [sigma, gamma] = ndgrid(params{:});
    params = [sigma(:) gamma(:)];
    p=size(params,1);
    
    %pre-allocate results arrays for speed
    Kern_CVerror = zeros(numFolds,1);
    CVmse = zeros(p,1);
    s= zeros(p,1);
    g= zeros(p,1);
    
    %for each parameter combination
    for i = 1:p
         %run cross validation for kernel ridge regression
        for f = 1:numFolds;
                
               %construct valdiation and training set 
                val = CVin(f,:);
                train = reshape(CVin (1:end ~= f,:),[1, trainSize- trainSize/numFolds]);
                
                X_CVval = X_train(val,:);   
                X_CVtrain = X_train(train,:);
                y_CVval = y_train(val);   
                y_CVtrain = y_train(train);
                
                 %compute kernel of full data for efficiency
                K = gaussKernel(X_train, params(i,1));
                K_trtr = K(train, train);
                K_tstr = K(val, train);
                
                 %run kernel ridge regression
                dual = kRidgeReg(K_trtr, y_CVtrain, params(i,2));
                
                %compute cross validation error
                Kern_CVerror(f) = dualCost(K_tstr, y_CVval, dual);
        end
        
        %store results after each cross validation run
        CVmse(i) = mean(Kern_CVerror);
        s(i) = params(i,1);
        g(i) = params(i,2);    
        
    
    end
        %store best hyper parameter pair
        minCVmse = min(CVmse);
        Sigma = s(CVmse== minCVmse);
        Gamma = g(CVmse== minCVmse);
        
end