

function [RR,  bestParam] = RRCVTuning(X, y, trainSize, numFolds, reg)

% RRCVTuning exectues and tunes ridge regression for a range of possible
% regularisation parameters (reg) using cross validation 
%
% Exepcts the additional mandatory inputs:
%   X= matrix of predictors
%   Y= corresponding vector of responses (must be same length as X_train columns)
%   trainSize = training data split size
%   numFolds = numboer fold cross validation folds to split training data
%   into for tuning

% Produces outputs:
%   RR = structured data array containing the regularisaiton parameters and
%   correspinding train, cross validaiton and test MSEs
%   bestParam = best regularisation parameter

        y_train = y(1 :trainSize); 
        y_test = y(trainSize+1:end); 
        X_train = X(1 :trainSize,:); 
        X_test = X(trainSize+1:end,:);
        
        %call method for giving matrix of randomly generated data indices for 
        %each cross validation fold - see CoreMethods for more details
    
        CVmse = 1:numFolds;
        CVin = CV(X_train, numFolds);
        
         %for each parameter 
        for i = 1:length(reg) 
           
           %run cross validation for  ridge regression
           for j = 1:numFolds;
                
                %construct valdiation and training set 
                  
                val = CVin (j,:);
                train = reshape(CVin (1:end ~= j,:),[1, trainSize- trainSize/numFolds]);
                
                X_CVval = X_train(val,:);   
                X_CVtrain = X_train(train,:);
                y_CVval = y_train(val);   
                y_CVtrain = y_train(train);
                
                %train ridge regression model
                w_est = RidgeR(X_CVtrain, y_CVtrain, reg(i));
                
                %call function to compute CV MSE
                CVmse(j) = meanSqError(w_est, X_CVval, y_CVval);                
           end
           
             %store results after each cross validation run
            RR.reg(i) = reg(i);
            RR.CVmse(i) = mean(CVmse);
            RR.trainmse(i)  = meanSqError(w_est, X_train, y_train);
            RR.testmse(i)  = meanSqError(w_est, X_test, y_test);
           
        end
        
        %store best reg parameter
        minCVmse = min([RR.CVmse]);
        bestParam = RR.reg([RR.CVmse]== minCVmse);
        
         %re- train ridge regression model on full training data, using the
         %best reg parameter
        w_est = RidgeR(X_train, y_train, bestParam);
        
         %call function to compute CV MSE and store
        RR.bestTestmse  = meanSqError(w_est, X_test, y_test);
end



