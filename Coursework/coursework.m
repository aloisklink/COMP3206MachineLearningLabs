%% Machine Learning Coursework

%% Creating Data
class(1).mean = [0; 3];
class(1).covar = [2 1; 1 2];
class(1).target = 1;
% creates the display style for graphing
class(1).style.Color = 'b';
class(1).style.LineStyle = 'none';
class(1).style.Marker = '.';
class(1).style.DisplayName = '\omega_1';

class(2).mean = [2; 1];
class(2).covar = [1 0; 0 1];
class(2).target = -1;
class(2).style.Color = 'r';
class(2).style.LineStyle = 'none';
class(2).style.Marker = '.';
class(2).style.DisplayName = '\omega_2';

valuesSize = 100;
% creates 100 samples from each class
for i = 1:length(class)
	class(i).in = mvnrnd(class(i).mean, class(i).covar, valuesSize);
	class(i).out = class(i).target * ones(size(class(i).in,1),1);
end

% combines the samples into one dataset
data.in = [class(1).in; class(2).in];
data.out = [class(1).out; class(2).out];

%% Posterior Probability and Baye's Optimal Decision Boundary
% Shows $ P (\omega_1 | \mathbf{x}) $, i.e. the probability that point $
% mathbf{x} $ belongs to class $ \omega_1 $.
classDistribution = gmdistribution( [class(1).mean'; class(2).mean'], ... 
	cat(3, class(1).covar, class(2).covar) );
clear X;
X(:,1) = -5:0.2:7.5; % the range to plot the graph on 
Y = X(:,1);
postProb = zeros(length(Y));
for i = 1:size(Y,1)	
	clear y; y(1:length(X),1) = Y(i);
	posteriorProb = posterior(classDistribution, [X y]);
	postProb(i,:) = posteriorProb(:,1);
end

% calculates the posterior probability for each of the data samples
class(1).post = posterior(classDistribution, [class(1).in(:,1) class(1).in(:,2)]);
class(2).post = posterior(classDistribution, [class(2).in(:,1) class(2).in(:,2)]);

% 3D plot of Posterior probability
figure(1);
% plots decision line at 50%
contour3(X, Y, postProb, [0.5 0.5],'r', 'DisplayName', 'Bayes');
hold on;
% plots the data samples
for cl = class
	plot3(cl.in(:,1), cl.in(:,2), cl.post(:,1), cl.style);
end
legend('show', 'Location', 'NorthEast');
% plots the posterior probability
surf(X, Y, postProb);
% removes the ugly black lines from the surf
shading flat;

view(14, 25);
xlabel('X_1'); ylabel('X_2'); zlabel('Posterior Probability');
title('Posterior Probability of \omega_1');
hold off;

% 2D Plot of the data samples and the decision boundary
figure(2); clf; hold on;
% plots the optimal decision boundary
contour(X, Y, postProb, [0.5 0.5],'g', 'DisplayName', 'Bayes Decision Boundary');
% plots the data
for cl = class
	plot(cl.in(:,1), cl.in(:,2), cl.style);
end
% legend('Bayes', '\omega_1', '\omega_2', 'Location', 'NorthEast');
legend('show');
xlabel('X_1'); ylabel('X_2');
title('100 Values of \omega_1 and \omega_2');
hold off;

%% Neural Network
neuralNet(1).numberOfLayers = 3;
neuralNet(2).numberOfLayers = 20;

% the style of the decision line:
neuralNet(1).contourStyle.LineColor = 'm'; % magenta
neuralNet(2).contourStyle.LineColor = 'c'; % cyan

for nNet = neuralNet
	nNet.contourStyle.DisplayName = [num2str(nNet.numberOfLayers) ' Layer Net Boundary'];
	
	[net] = feedforwardnet(nNet.numberOfLayers);
	[net] = configure(net, data.in', data.out');
	[net] = train(net, data.in', data.out');

	netOutput = zeros(length(Y));
	for j = 1:length(Y)	
		clear y; y(1:length(X),1) = Y(j);
		netOutput(j,:) = net([X y]');
	end

	% creates a plot for the decision boundary
	figure(3+i);
	% draws the decision boundary
	contour3(X, Y, netOutput, [0.0 0.0], 'r', 'DisplayName', 'Decision Boundary');
	hold on;
	% removes the ugly black lines from the surf
	for cl = [class(1) class(2)]
		plot3(cl.in(:,1), cl.in(:,2), net([cl.in(:,1) cl.in(:,2)]'), cl.style);
	end
	legend('show','Location', 'Best');
	% draws the output of the neural net
	surf(X, Y, netOutput); 
	shading flat;
	xlabel('X_1'); ylabel('X_2'); zlabel('Output Weights (\omega_1 is positive)');
	title([num2str(nNet.numberOfLayers) ' Layer Neural Network']);
	view(-14, 25);
	hold off;
	
	% adds contour lines to 2D plot
	figure(2);
	hold on;
	contour(X, Y, netOutput, [0.0 0.0], 'LineColor', nNet.contourStyle.LineColor, ...
		'DisplayName', nNet.contourStyle.DisplayName);
	hold off;
end
legend('off'); legend('show','Location', 'Best');

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