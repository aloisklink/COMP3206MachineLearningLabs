%% Time Series Prediction
[samples, times] = createMackeySamples(2000);
p = 20;
% create training data
train_data_max = 1500;
train_data_min = 1;

train_data_actual_min = train_data_min + p;
train_data_size = train_data_max-train_data_actual_min;
% input 
train_data.in = zeros(train_data_size, p);
train_data.out = zeros(train_data_size, 1);
j = 1;
for i = train_data_actual_min:train_data_max
	% adds 1 for offset
	train_data.in(j,:) = samples( (i - p): (i - 1));
	train_data.out(j) = samples(i);
	train_data.time(j) = times(i);
	j = j + 1;
end
% create testing data
test_data_max = 2000;
test_data_min = train_data_max + 1;
test_data_size = test_data_max - test_data_min;
test_data.in = zeros(test_data_size, p);
test_data.out = zeros(test_data_size, 1);
j = 1;
for i = test_data_min:test_data_max
	% adds 1 for offset
	test_data.in(j,:) = samples( (i - p): (i - 1));
	test_data.out(j) = samples(i);
	test_data.time(j) = times(i);
	j = j + 1;
end

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