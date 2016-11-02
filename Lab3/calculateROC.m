function [ ROC, areaUnderROC ] = calculateROC( p1, p2, rocResolution )
%calculateROC Calculates the ROC of p1 and p2
%   Calculates the Reciever Operating Characteristic curve of p1 and p2, by
%   sliding a decsision threshold
	[~, xx1] = hist(p1);
	[~, xx2] = hist(p2);
	thmin = min([xx1 xx2]);
	thmax = max([xx1 xx2]);
	thRange = linspace(thmin, thmax, rocResolution);
	
	ROC = zeros(rocResolution,3);
	
	for jThreshold = 1: rocResolution
		threshold = thRange(jThreshold);
		tPos = length(find(p1 > threshold))*100 / size(p1,1);
		fPos = length(find(p2 > threshold))*100 / size(p2,1);
		ROC(jThreshold,:) = [fPos tPos threshold];
	end
	
	correctDirectionROC = sortrows(ROC,1);%% Sorting ROC to start from 0%.
	areaUnderROC = trapz(correctDirectionROC(:,1), correctDirectionROC(:,2));
end

