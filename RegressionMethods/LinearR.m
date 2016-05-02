
function w = LinearR(X, y)
% LinearR computes the linear regression model coeficient vector, given a
% training matrix X, and an equal length response vector y.

    w = ((X'*X)\X'*y);
end



