%% Machine Learning Coursework
class(2).mean = [0; 3];
class(2).covar = [2 1; 1 2];
class(1).mean = [2; 1];
class(1).covar = [1 0; 0 1];

valuesSize = 100;
X1 = mvnrnd(class(1).mean, class(1).covar, valuesSize);
X2 = mvnrnd(class(2).mean, class(2).covar, valuesSize);

%% Posterior Probability
% Shows $ P (\omega_1 | \mathbf{x}) $, i.e. the probability that point $
% mathbf{x} $ belongs to class $ \omega_1 $.
classDistribution = gmdistribution([class(1).mean'; class(2).mean'], cat(3, class(1).covar, class(2).covar));
clear X; clear y;
X(:,1) = -5:0.2:7; 
Y = X(:,1);
postProb = zeros(length(Y));
for i = 1:size(Y,1)	
	clear y; y(1:size(X,1),1) = Y(i);
	posteriorProb = posterior(classDistribution, [ X(:,1) y(:,1)]);
	postProb(i,:) = posteriorProb(:,1);
end

% calculates the posterior probability for each of the data
X1Post = posterior(classDistribution, [X1(:,1) X1(:,2)]);
X2Post = posterior(classDistribution, [X2(:,1) X2(:,2)]);

% 3D plot
figure(1);
% plots line at 50%
contour3(X, Y, postProb, [0.5 0.5],'r');
hold on;
% plots the data
plot3(X1(:,1), X1(:,2), X1Post(:,1), 'b.');
plot3(X2(:,1), X2(:,2), X2Post(:,1), 'r.');
% plots the posterior probability
surf(X, Y, postProb);
% removes the ugly black lines from the surf
shading flat;

view(-111, 21);
legend('Bayes', '\omega_1', '\omega_2', 'Location', 'NorthEast');
xlabel('X_1'); ylabel('X_2'); zlabel('Posterior Probability');
title('Posterior Probability of \omega_1');
hold off;

% 2D Plot
figure(2); clf; hold on;
% plots the optimal decision boundary
contour(X, Y, postProb, [0.5 0.5],'g');
% plots the data
plot(X1(:,1), X1(:,2),'b.');
plot(X2(:,1), X2(:,2),'r.');

legend('Bayes', '\omega_1', '\omega_2', 'Location', 'NorthEast');
xlabel('X_1'); ylabel('X_2');
title('100 Values of \omega_1 and \omega_2');
hold off;