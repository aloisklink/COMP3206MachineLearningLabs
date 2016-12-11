%% Chaotic Time Series Prediction
% Using the Mackey-Glass Time Series

% creates the time series
[samples, times] = createMackeySamples(2000);

% converts the time series to input/output for machine learning
training_range = 1:1500;
testing_range = 1501:2000;
p = 20; % number of past data to use as input
[train_data, test_data] = ...
	convertTimeSeriesToMLInput(samples, times, p, training_range, testing_range);

%% Linear Regression
% Least squares regression as pseudo inverse
leastSquareReg = @(Ytr, Ftr, Yinput) ...
    [Yinput ones(length(Yinput),1)]* ...
	( ([Ytr ones(length(Ytr),1)]'*[Ytr ones(length(Ytr),1)]) \ ...
	   [Ytr ones(length(Ytr),1)]'*Ftr);

test_data_predicted = leastSquareReg(train_data.in, train_data.out, test_data.in);

squareError = @(Factual, Fpredict) ...
	(Factual - Fpredict).'*(Factual-Fpredict);
disp(['Square Error of Linear Regression Prediction is: ' num2str(squareError(test_data.out, test_data_predicted))]);

figure(1);
plot(test_data.time, test_data.out, 'b--', 'DisplayName', 'Expected Data');
hold on;
plot(test_data.time, test_data_predicted, 'r-.', 'DisplayName', 'Linear Regression Prediction');
legend('show');
title('Mackey-Glass Time Series Prediction');
xlabel('Time');
hold off;

%% Neural Network

[net] = feedforwardnet(10);
% must transpose everything for the neural network
[net] = configure(net, train_data.in', train_data.out');
[net] = train(net, train_data.in', train_data.out');

test_data_nNet = net(test_data.in');
figure(1);
hold on;
plot(test_data.time, test_data_nNet', 'c-.', 'DisplayName', 'Neural Net Prediction');
legend('off'); legend('show');
hold off;

disp(['Square Error of Neural Network Prediction is: ' num2str(squareError(test_data.out, test_data_nNet'))]);
	