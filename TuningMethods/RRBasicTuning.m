
function [RR, bestParam] = RRBasicTuning(X, y, trainSize, valSize, reg)
% RRBasicTuning exectues and tunes ridge regression for a range of possible
% regularisation parameters (reg), using training mse or validation mse.
%
% Exepcts the additional mandatory inputs:
%   X= matrix of predictors
%   Y= corresponding vector of responses (must be same length as X_train columns)
%   trainSize = training data split size
%   valSize = how much of training data to attribute to a validation set 

% NOTE: If valSize is set to 0, then the best regularisation parameter is
% chosen by miniming training data only. Else, the best regularisation
% parmater is chosen by minimising the validation error.

% Produces outputs:
%   RR = structured data array containing the regularisaiton parameters and
%   correspinding train, validaiton and test MSEs
%   bestParam = best regularisation parameter
        
        tS = trainSize; 
        vS = round(tS*valSize);
        
        y_val = y(1:vS); 
        y_train = y(vS +1 :tS); 
        y_test = y(tS+1:end); 
        X_val =  X(1:vS,:);
        X_train = X(vS +1 :tS,:); 
        X_test = X(tS+1:end,:);

        for i = 1:size(reg,2)   

            RR.reg(i) = reg(i);
            
            %Execute ridge regression for the current regularisation
            %parameter
            RR.w_est(:,i) = RidgeR(X_train, y_train, reg(i)); 
            
            %Compute training, validation(if applicable) and test MSE and
            %store to results array
            RR.trainmse(i) = meanSqError(RR.w_est(:,i), X_train , y_train);
            if valSize ~= 0
                RR.valmse(i) = meanSqError(RR.w_est(:,i), X_val, y_val);
            end
            RR.testmse(i) = meanSqError(RR.w_est(:,i), X_test, y_test);
            
        end
        
       minTrainmse = min([RR.trainmse]);
       bestParam= RR.reg([RR.trainmse] == minTrainmse);
       if valSize ~= 0
           minValmse = min([RR.valmse]);
           bestParam =  RR.reg([RR.valmse] == minValmse); %Best param determined by Val
       end
       
       %re-train ridge regression model using the best chosen
       %regularisation parameter.
       
       w_est = RidgeR(X_train, y_train, bestParam);
       RR.bestTestmse  = meanSqError(w_est, X_test, y_test);
end

        
        