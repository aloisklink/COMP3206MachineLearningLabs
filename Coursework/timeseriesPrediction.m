%% Time Series Prediction
[samples, times] = createMackeySamples(2000);
p = 20;

% converts the time series to input/output for machine learning
training_range = 1:1500;
testing_range = 1501:2000;
[train_data, test_data] = ...
	convertTimeSeriesToMLInput(samples, times, p, training_range, testing_range);

% Least squares regression as pseudo inverse
leastSquareReg = @(Ytr, Ftr, Yinput) ...
    [Yinput ones(length(Yinput),1)]* ...
	( ([Ytr ones(length(Ytr),1)]'*[Ytr ones(length(Ytr),1)]) \ ...
	   [Ytr ones(length(Ytr),1)]'*Ftr);

test_data_predicted = leastSquareReg(train_data.in, train_data.out, test_data.in);

squareError = @(Factual, Fpredict) ...
	(Factual - Fpredict).'*(Factual-Fpredict);
disp(['Square Error is: ' num2str(squareError(test_data.out, test_data_predicted))]);

plot(test_data.time, test_data.out, test_data.time, test_data_predicted);