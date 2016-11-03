%% Lab Three
%% Create Data and Plot Contours
% Creates $ X \in \omega_i \sim \mathcal{N}_2 (\mathbf{m_i}, \mathbf{C_i})$
% where $ \mathbf{m_1} = [0 2]^t $, $ \mathbf{m_2} = [1.7 2.5]^t $, and
% $ \mathbf{C_1} = \mathbf{C_2} = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix} $
m1 = [0 2]';
m2 = [1.7 2.5]';
sigma = [2 1; 1 2];
C1 = sigma;
C2 = sigma;
numGrid = 50;

xRange = linspace(-6.0, 6.0, numGrid);
yRange = linspace(-6.0, 6.0, numGrid);

P1 = zeros(numGrid, numGrid);
P2 = P1;
for i=1:numGrid
	for j=1:numGrid;
	x = [yRange(j) xRange(i)]';
	P1(i,j) = mvnpdf(x', m1', C1);
	P2(i,j) = mvnpdf(x', m2', C2);
	end
end

Pmax = max(max([P1 P2]));

figure(1), clf,
% drawing contour lines at P = 10%, 50%, 80%
contour(xRange.', yRange, P1, [0.1*Pmax 0.5*Pmax 0.8*Pmax], 'LineWidth', 2);
hold on
plot(m1(1), m1(2), 'b*', 'LineWidth', 4);
contour(xRange.', yRange, P2, [0.1*Pmax 0.5*Pmax 0.8*Pmax], 'LineWidth', 2);
plot(m2(1), m2(2), 'r*', 'LineWidth', 4);
%% Draw Data
N = 200;
X1 = mvnrnd(m1, C1, N);
X2 = mvnrnd(m2, C2, N);
plot(X1(:,1),X1(:,2),'bx', X2(:,1),X2(:,2),'ro');grid on
%% Fischer Linear Discriminant
% The Fischer Linear Disciminant is the direction on which projected data
% is maximally separable, which can be calculated from:
%
% $$\mathbf{w}_F = \alpha \mathbf{C}_W ^-1 (\mathbf{m}_1 - \mathbf{m}_2)$$

wF = (C1+C2)\(m1-m2);
xx = -6:0.1:6;
yy = xx*wF(2)/wF(1);
plot(xx,yy,'r', 'LineWidth', 2);
%% Histograms of Projected Data
% Projects the data onto the Fischer Direction and plots a histogram.
p1 = X1*wF;
p2 = X2*wF;
plo = min([p1; p2]);
phi = max([p1; p2]);
[nn1, xx1] = hist(p1);
[nn2, xx2] = hist(p2);
hhi = max([nn1 nn2]);
figure(2),
subplot(211), bar(xx1, nn1);
axis([plo phi 0 hhi]);
title('Distribution of Projections', 'FontSize', 16)
ylabel('Class 1', 'FontSize', 14)
subplot(212), bar(xx2, nn2);
axis([plo phi 0 hhi])
ylabel('Class 2', 'FontSize', 14)
%% ROC Curve
% Creates a Reciceiver Operating Characteristic (ROC) Curve that shows True
% Positive (TP) and False Positive (FP) rates.
[ROC, areaROC] = calculateROC( p1, p2, 50);
figure(3), clf, plot(ROC(:,1), ROC(:,2), 'b', 'LineWidth', 2);
axis([0 100 0 100]); grid on, hold on; plot(0:100, 0:100, 'b-');
xlabel('False Positive'); ylabel('True Positive');
title('Receiver Operating Characteristic Curve');
%% ROC Curve Area
disp(['The area of the ROC is ' num2str(areaROC)]);
%% Classification Accuracy
% Computes the classification accuracy for the True Positive rate of $ 70% $.
truePositiveToFind = 70;
[minValue, closestIndex] = min( abs( ROC(:,2) - truePositiveToFind) );
threshold = ROC(closestIndex, 3);
tPos = length(find(p1 > threshold))*100 / N;
fPos = length(find(p2 > threshold))*100 / N;
%% More ROC Curves
% Shows the ROC Curves for the data projected onto a random direction and
% onto the direction connecting the two means.
randAng = rand() * 2* pi; %random direction
randomW = [sin(randAng) cos(randAng)]'; %random direction
[randROC, randROCArea] = calculateROC( X1*randomW, X2*randomW, 50);
disp(['The area of the RandROC is ' num2str(randROCArea)]);

betweenMeanW = m1 - m2;
[betMROC, betMROCArea] = calculateROC( X1*betweenMeanW, X2*betweenMeanW, 50);
disp(['The area of the betweenMeansROC is ' num2str(betMROCArea)]);

figure(4), clf, plot(randROC(:,1), randROC(:,2), 'b', 'LineWidth', 2); hold on;
plot(betMROC(:,1),betMROC(:,2)); legend('Random Direction ROC', 'Between Mean ROC');
axis([0 100 0 100]); grid on; plot(0:100, 0:100, 'b-');
xlabel('False Positive'); ylabel('True Positive');
title('Receiver Operating Characteristic Curve'); hold off;
%% Nearest-Neighbour Classifier
% The Nearest-Neighbour Classifier finds the closest training point to the
% test point, and use its class to classify the test data.
X = [X1; X2];
N1 = size(X1, 1);
N2 = size(X2, 1);
y = [ones(N1,1); -1*ones(N2,1)];
d = zeros(N1+N2-1,1);
nCorrect = 0;
for jtst = 1:(N1+N2)
	% pick a point to test
	xtst = X(jtst,:);
	ytst = y(jtst);
	% All others form the training set
	jtr = setdiff(1:N1+N2, jtst);
	Xtr = X(jtr,:);
	ytr = y(jtr,1);
	% Compute all distances from test to training points
	for i=1:(N1+N2-1)
		d(i) = norm(Xtr(i,:)-xtst);
	end
	% Which one is the closest?
	[imin] = find(d == min(d));
	% Does the nearest point have the same class label?
	if ( ytr(imin(1)) * ytst > 0 )
		nCorrect = nCorrect + 1;
	end
end

% Percentage correct
pCorrect = nCorrect*100/(N1+N2);
disp(['Nearest neighbour accuracy: ' num2str(pCorrect) '%']);
%% Distance-to-Mean Classifiers
% The following code shows two Distance-to-Mean Classifiers, which classify
% test data depending on the distance between it and the means of the
% training data. The Euclidean Distance classifier uses the normal 
% $ distance = \sqrt{\sum_{i=1}^n (q_i-p_i)^2} $ formula. However, the 
% /href{https://en.wikipedia.org/wiki/Mahalanobis_distance}{Mahalanobis distance}
% takes into account variance and correlation, thus being a bit measure of
% distance.
for jtst = 1:(N1+N2)
	% pick a point to test
	xtst = X(jtst,:);
	ytst = y(jtst);
	% All others form the training set
	jtr = setdiff(1:N1+N2, jtst);
	Xtr = X(jtr,:);
	ytr = y(jtr,1);
	% find means and euclidian distances
	mahDistance1 = mean(Xtr((ytr > 0),:)) - xtst;
	mahDistance2 = mean(Xtr((ytr < 0),:)) - xtst;
	% compare distances
	eucDistance = @(x) sqrt(x(1)^2 + x(2)^2);
	if eucDistance(mahDistance1) < eucDistance(mahDistance2)
		yVal = 1;
	else
		yVal = -1;
	end
	% Does the nearest point have the same class label?
	if yVal * ytst > 0
		eucCorrect = eucCorrect + 1;
	end
	
	% find Mahalanobis distances
	mahDistance1 = mahal(xtst, Xtr(ytr > 0,:));
	mahDistance2 = mahal(xtst, Xtr(ytr < 0,:));
	% compare distances
	if mahDistance1 < mahDistance2
		yVal = 1;
	else
		yVal = -1;
	end
	% Does the nearest point have the same class label?
	if yVal * ytst > 0
		mahCorrect = mahCorrect + 1;
	end
end	
% Percentage correct
eucPCorrect = eucCorrect*100/(N1+N2);
disp(['Euclidiean Distance-to-mean accuracy: ' num2str(eucPCorrect) '%']);
mahPCorrect = mahCorrect*100/(N1+N2);
disp(['Mahalanobis Distance-to-mean accuracy: ' num2str(mahPCorrect) '%']);
%% Posterior Probability
% Shows $ P (\omega_1 | \mathbf{x}) $, i.e. the probability that point $
% mathbf{x} belongs to class $ \omega_1 $.
classDistribution = gmdistribution([m1'; m2'], cat(3, C1, C2));
clear X; clear y;
X(:,1) = -5:0.1:5; Y = X(:,1);
y(1:size(X,1),1) = Y(1);
postProb = posterior(classDistribution, [ X(:,1) y(:,1)]);
postProb2 = postProb(:,2);
for i = 2:size(Y,1)	
	clear y; y(1:size(X,1),1) = Y(i);
	postProb = posterior(classDistribution, [ X(:,1) y(:,1)]);
	postProb2 = [postProb2 postProb(:,2)];
end
surf(X, Y, postProb2); xlabel('X_1'); ylabel('X_2'); zlabel('$P (\omega_1 | \mathbf{x})$');
title('Posterior Probability');
%% Baye's Optimal Class Boundary
% Calculates and shows the Baye's Optimal Class Boundary if both the means
% $ \mathbf{m}_i $
% and the covariances $ \mathbf{C}_i $ are different.
% As the Baye's Classifier Decision Rule is:
% $$ f(\mathbf{x}) = \begin{cases} \omega_1, & (P (\omega_1 | \mathbf{x}) >
% P (\omega_2 | \mathbf{x}) \\ \omega_2, & (P (\omega_1 | \mathbf{x}) <
% P (\omega_2 | \mathbf{x}) \\ \end{cases} $$, and $$ P (\omega_i |
% \mathbf{x}) = \frac{1}{c} P (\mathbf{x} | \omega_i) P(\omega_i) $$ and $$ P (\mathbf{x} | \omega_i)
% = \mathcal{N}(\boldsymbol m_i, \boldsymbol C_i) = \frac{1}{\sqrt{(2\pi)^2|C|}
% \exp(-\frac{1}{2} (\mathbf{x} - \mathbf{m}_i)^T \mathbf{C}_i^{-1}
% (\mathbf{x} - \mathbf{m}_i) ) $$
C2 = [1.5 0; 0 1.5];
X2 = mvnrnd(m2, C2, N);
plot(X1(:,1),X1(:,2),'bx', X2(:,1),X2(:,2),'ro');grid on;hold on;

normalProb = @(x, mean, covar) ((1/( ((2*pi)^(size(mean,1)/2))*det(covar)^(1/2))) * exp(-1/2*(x'-mean)'/(covar)*(x'-mean)));
varX = sym('x', [1 2], 'real');
eqn = normalProb(varX, m1, C1)*.5 == normalProb(varX, m2, C2)*.5;
[solx, ~, ~, ~] = solve(eqn,varX, 'ReturnConditions', true);
clear x
x(1,:) = -10:0.123:10; x(2,:) = subs(solx, x(1,:)); axis([-6 6 -6 6]);
plot(x(1,:), x(2,:), 'DisplayName', 'Bayes Optimal Class Boundary');