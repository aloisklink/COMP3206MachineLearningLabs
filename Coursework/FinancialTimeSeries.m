%% Financial Time Series

%loads row 2, column 2 onwards (row 1 is title, column 1 is string dates)
SnP500 = csvread('SnP500.csv', 1, 1);
ClosingPrice = 4; Volume = 5;
% file is in reverse chronological order
samplesReverse = SnP500(:,ClosingPrice);
samples = flipud(samplesReverse);
size = length(samples);
times = 1:size;

% converts the time series to input/output for machine learning
training_range = 1:round(size*0.75);
testing_range = (round(size*0.75)+1):size;
p = 20; % number of past data to use as input
[train_data, test_data] = ...
	convertTimeSeriesToMLInput(samples, times, p, training_range, testing_range);
% gets Volume Traded data
[trainVolData, testVolData] = ...
	convertTimeSeriesToMLInput(flipud(SnP500(:,Volume)), times, p, training_range, testing_range);

% creates a multi-variable input: 20 closing prices, 20 volume data
train_data.in = [train_data.in trainVolData.in];
test_data.in = [test_data.in testVolData.in];
nets(1).neurons = 20;
nets(2).neurons = 20;
nets(1).lineOptions = 'm--';
nets(2).lineOptions = 'b-.';

% plots the expected closing price
figure(1);
plot(test_data.time, test_data.out, 'r:', 'DisplayName', 'Expected Data', ...
	'LineWidth', 1.5);
hold on;
legend('show');
title('S&P 500 Prediction');
xlabel('Days since 2011-12-11'); ylabel('Closing Price in USD');
% i = number of input variables
for i = 1:2
	% create feedforward net with loads of layers
	% these layers aren't necessary for one-step-ahead, but they make the 
	% free-running code more stable
	[net] = cascadeforwardnet(nets(i).neurons);
	% must transpose everything for the neural network
	% trains the net with the first p*i variables
	[net] = train(net, train_data.in(:,1:p*i)', train_data.out');

	% tests the net
	test_data_nNet = net(test_data.in(:,1:p*i)');
	
	figure(1); hold on;
	% plots the test prediction
	plot(test_data.time, test_data_nNet', nets(i).lineOptions, ...
		'DisplayName', ['Neural Net Prediction ' num2str(i) ' Input Variables']);
	legend('off'); legend('show', 'Location', 'Best');
	hold off;

	squareError = @(Factual, Fpredict) ...
		(Factual - Fpredict).'*(Factual-Fpredict);
	disp(['Square Error of Neural Network Prediction from ' num2str(i) ' input variables: ' ...
		num2str(squareError(test_data.out, test_data_nNet'))]);

	% finds the average % gain per day
	GainPerDay = zeros(20,1);
	for trial = 1:length(GainPerDay)
		[net] = cascadeforwardnet(nets(i).neurons);
		% must transpose everything for the neural network
		[net] = configure(net, train_data.in(:,1:p*i)', train_data.out');

		[net] = train(net, train_data.in(:,1:p*i)', train_data.out');

		test_data_nNet = net(test_data.in(:,1:p*i)');
		% buy if difference is positive, ie price is going up
		% sell if difference is negative, ie price is going down
		buyOrSell = (diff(test_data_nNet) > 0) - (diff(test_data_nNet) < 0);
		% if buy and price goes up/sell and price goes down, profit!
		% else lose money
		GainPerDay(trial) = mean( diff(test_data.out') .* buyOrSell / test_data.out(1:length(test_data.out)-1)');
	end
	% shows the average average gain per day
	disp(['Gain Per Day from ' num2str(i) ' input variables: ' num2str(mean(GainPerDay))]);
end
