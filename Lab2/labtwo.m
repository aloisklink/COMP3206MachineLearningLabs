%% COMP3206/COMP6229(2016/17): Machine Learning Lab 2
%
%% Part 1: Finding the Perpendicular Distance to Origin
% The following code creates a line, finds the angle of the line, then the
% perpendicular angle, and finally use the perpendicular angle and the
% distance given by the equation $-c/\sqrt{a^2_1+b^2_1}$ to create a line
% from the origin, which should meet the original line perfectly.
plot([0 -4], [-3 0], 'b', 'LineWidth', 2, 'DisplayName', 'Original Line'); 
title('Finding the Perpendicular Distance to Origin');
axis([-5 1 -5 1]); grid on;
hold on;
perp(1, :) = -5:0.1:5;
a = 3;
b = 4;
c = 12;
gradient = a/-b;

%distance equation given to us
distance = -c/sqrt(a^2 + b^2);

%calculating the angle with trigonometry
angle(1) = atan(gradient(1));
%calculating the perpendicular angle by adding pi/2
angle(2) = angle(1) + pi/2;

% draws a line with the perpendicular angle and distance given
plot([0 distance*cos(angle(2))], [0 distance*sin(angle(2))], 'DisplayName','Perpendicular Distance to Origin');
legend('show'); hold off;

%%
% As can be seen in [INSERT ABOVE], the above line is perpendicular with
% the the original line, goes to the origin, is the same length as the
% distance given, and meets the line.

%% Part 2: Generating Samples from Bi-Variate Normal Densities with Distinct Means
means(1,:) = [0   2].';
means(2,:) = [1.5 0].';
sigma = [2 1; 1 2];
valuesSize = 100;

X1 = mvnrnd(means(1, :).', sigma, valuesSize);
X2 = mvnrnd(means(2, :).', sigma, valuesSize);

plot(X1(:,1),X1(:,2),'r.', 'DisplayName', 'Class +1');
hold on; title('Data and Class Boundaries');
plot(X2(:,1),X2(:,2),'mx', 'DisplayName', 'Class -1');
axis([-5 5 -5 5]);
%% Part 3: Computing Bayes Optimal Class Boundry
% Solves the solutions for the equation $\mathbf{\omega^T \times X} - 1 = 0 $ 
invSig = inv(sigma);
w = 2*invSig*(means(1,:).' - means(2,:).');
b = ( transpose(means(1,:).')*invSig*means(1,:).' - ...
    ( transpose(means(2,:).')*invSig*means(2,:).')) - ...
    log(1);
varX = sym('x', [2 1]);
eqn = transpose(w)*varX + b == 0;
[solx, ~, ~, ~] = solve(eqn,varX, 'ReturnConditions', true);
x(1,:) = [-5 5]; x(2,:) = subs(solx, x(1,:));
plot(x(1,:), x(2,:), 'DisplayName', 'Bayes Optimal Class Boundary');
%% Part 4: Perceptron Learning Algorithm
data = [X1 ones(valuesSize,1) -1*ones(valuesSize,1)];
data = [data; X2 ones(valuesSize,1) 1*ones(valuesSize,1)];
valuesSize = size(data);
selection = randperm(valuesSize(1));
trainingPercentage = 0.8;

% seperating data into training and test data
trainingData = data(selection(1:valuesSize(1)*trainingPercentage),:);
testData = data(selection(valuesSize(1)*trainingPercentage+1:valuesSize(1)),:);

weights = zeros(1,3);
eta = 0.001;
trainingDataSize = size(trainingData);
for iter=1:10000
	j = randi([1 trainingDataSize(1)]);
	result = dot(trainingData(j,1:3),weights);
	if result >= 0
		result = 1;
	else
		result = -1;
	end
	% divided by 2 because we want error to be 1, 0, or -1.
	error = (trainingData(j,4) - result)/2;
	weights = weights + eta * error * trainingData(j,1:3);
end
testDataSize = size(testData);
success = 0;
for iter=1:testDataSize(1)
	result = dot(testData(iter,1:3),weights);
	if result * testData(iter, 4) > 0
		success = success + 1;
	end
end
SuccessRatio = success/testDataSize(1);
disp(['Success Ratio =' num2str(SuccessRatio)]);

input = sym('X', [2 1]); varIn = [input; 1];
eqn = dot(varIn,weights) == 0;
[solx, ~, ~, ~] = solve(eqn,input, 'ReturnConditions', true);
x(1,:) = [-5 5]; x(2,:) = subs(solx, x(1,:));
plot(x(1,:), x(2,:), 'r', 'DisplayName', 'Perceptron Boundary'); 
legend('show'); hold off;
%%
% As you can see in [INSERT FIGURE REFERENCE], the boundary of
% the Perceptron Learning Algorithm is quite similar to the Bayes Optimal
% Class Boundry, but not perfect. As eta and the max number of iterations
% are increased, the accuracy of the Perceptron Learning Algorithm
% increases, but it reaches a limit as it is impossible to classifly all
% the data correctly, due to the nature of a Normal Distribution having a
% range of infinity. 