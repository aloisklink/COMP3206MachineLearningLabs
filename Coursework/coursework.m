%% Machine Learning Coursework
class(2).mean = [0; 3];
class(2).covar = [2 1; 1 2];
class(1).mean = [2; 1];
class(1).covar = [1 0; 0 1];

valuesSize = 1000;
X1 = mvnrnd(class(1).mean, class(1).covar, valuesSize);
X2 = mvnrnd(class(2).mean, class(2).covar, valuesSize);

%% Posterior Probability
% Shows $ P (\omega_1 | \mathbf{x}) $, i.e. the probability that point $
% mathbf{x} $ belongs to class $ \omega_1 $.
classDistribution = gmdistribution([class(1).mean'; class(2).mean'], cat(3, class(1).covar, class(2).covar));
clear X; clear y;
X(:,1) = -5:0.1:5; Y = X(:,1);
y(1:size(X,1),1) = Y(1);
postProb2 = [];
for i = 1:size(Y,1)	
	clear y; y(1:size(X,1),1) = Y(i);
	postProb = posterior(classDistribution, [ X(:,1) y(:,1)]);
	postProb2 = [postProb2 postProb(:,2)];
end
figure(1);
surf(X, Y, postProb2); hold on;
clear Zs;
Zs(1:length(X1)) = 0.50;
contour3(X, Y, postProb2, [0.5 0.5],'r'); xlabel('X_1'); ylabel('X_2'); zlabel('Probability');
shading interp;
title('Posterior Probability');
hold off;
figure(2); clf; hold on;
plot(X1(:,1), X1(:,2),'r.');
plot(X2(:,1), X2(:,2),'bx');
contour(X, Y, postProb2, [0.5 0.5],'w:');
hold off;
