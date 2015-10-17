%% Obtain test set error

theta = trainLinearReg(X_poly, y, 3);
error_test = linearRegCostFunction(X_poly_test, ytest, theta, 0);