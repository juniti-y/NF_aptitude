%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% When applying this script to your own data, please replace the following
% variables between lines 18-20:
%  * FC_all: N-by-D dim. Functional connectivity data matrix, where the
%    rows are N study participants and the columns are the values of D
%    distinct FC.
%  * NFscore_all: N-dim. column vector containing the neurofeedback scores
%    of N study participants
%  * mask_train: N-dim. column vector containing dummy variables, each of
%    which indicates whether the participant should be used for the training
%    data (indicated by "true") or the test data (indicated by "false").
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear
close all

%generation of dummy data (Please replace them by your own data.)
FC_all=rand(65, 4)*2-1; %functional connectivity
NFscore_all=rand(65, 1); %NF aptitude score
mask_train=[true(41,1); false(24,1)]; %41 training data and 24 test data

X_train = FC_all(mask_train,:); %explanatory variable of training data (discovery dataset)
Y_train = NFscore_all(mask_train); %objective variable of training data (discovery dataset)
N_train = length(Y_train); %number of training data (discovery dataset)
X_test = FC_all(~mask_train,:); %explanatory variable of test data (independent dataset)
Y_test = NFscore_all(~mask_train); %objective variable of test data (independent dataset)

% Get the total number of FCs
Tot_Num_FC = size(FC_all, 2);

var_idx = cell(27000000,1);
idx = 0;
for p = 1:Tot_Num_FC %8 choose 2
    tmp = combnk(1:Tot_Num_FC,p);
    for k = 1:size(tmp,1)
        idx = idx+1;
        var_idx{idx} = tmp(k,:); %sets of FCs from all FCs
    end
end
var_idx(idx+1:end) = []; 
MaxK = length(var_idx);
Fitness = nan(1,MaxK); %store the result
for k = 1:MaxK
	if mod(k, 1000000)==0
	disp([k,MaxK]) %show progress
	end
    mask_var = false(1,Tot_Num_FC);
    mask_var(var_idx{k}) = true;
    Y_pred = nan(size(Y_train));

    %LOOCV to choose the best combination of FCs
    for n = 1:N_train 
        x1 = X_train(:,mask_var);  %explanatory variable of training data
        x1(n,:) = []; %remove 1 sample
        y1 = Y_train; %objective variable of training data
        y1(n) = []; %remove 1 sample
        [~,~,~,~,betaPLS] = plsregress(x1,y1,1); %training PLS model
        x0 = X_train(n,mask_var); %test data
        Y_pred(n) = [ones(size(x0,1),1), x0]*betaPLS; %prediction of PLS model for left 1 sample
    end
    [Fitness(k),~] = corr(Y_train, Y_pred, 'type', 'Spearman'); %correlation between true and predicted NF score
end
%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5

[~,max_index]=max(Fitness); %take best index
%[~,sort_Fitness]=sort(Fitness,'descend');
%for i_search=1:1000
%max_index=sort_Fitness(i_search);
var_best = var_idx{max_index}; %best combination of FCs

%evaluate the best model 
for n = 1:N_train
    x1 = X_train(:,var_best);
    x1(n,:) = [];
    y1 = Y_train;
    y1(n) = [];
    [~,~,~,~,betaPLS] = plsregress(x1,y1,1);
    x0 = X_train(n,var_best);
    Y_pred(n) = [ones(size(x0,1),1), x0]*betaPLS;
end

%scatter plot
figure;

co = get(gca,'colororder');
%clf(fig1);
scatter(Y_train, Y_pred, 100, co(1,:),'filled');
hold on;
mdl = polyfit(Y_train, Y_pred,1);
xrng = get(gca,'XLim');
x1 = min(xrng):0.01:max(xrng);
y1 = polyval(mdl,x1);
plot(x1,y1,'k--');
xlim(xrng);
axis square;
[rho, pval] = corr(Y_train, Y_pred, 'type', 'Spearman');
xlabel('True NF score');
ylabel('Predicted NF score');
title(sprintf('LOOCV: rho = %.3f (p = %.3f)', rho, pval));


 %AUC
C_train = Y_train > median(Y_train);
figure;
%clf(fig1);
[fpr,tpr,~,AUC] = perfcurve(C_train, Y_pred, 1);
plot(fpr,tpr,'Color',co(1,:));
hold on;
plot([0,1],[0,1],'k:');
axis square;
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title(sprintf('ROC Curve for validation (AUC = %.3f)',AUC));


%evaluate the best model using the independent dataset
[Xloadings,Yloadings,Xscores,Yscores,betaPLS] = plsregress(X_train(:,var_best),Y_train,1); %training PLS model
Y_pred = [ones(size(X_test(:,var_best) ,1),1) X_test(:,var_best)]*betaPLS; %prediction

%scatter plot
figure;
%clf(fig1);
scatter(Y_test, Y_pred, 100,co(2,:),'filled');
hold on;
mdl = polyfit(Y_test, Y_pred,1);
xrng = get(gca,'XLim');
x1 = min(xrng):0.01:max(xrng);
y1 = polyval(mdl,x1);
plot(x1,y1,'k--');
xlim(xrng);
axis square;
[rho, pval] = corr(Y_test, Y_pred, 'type', 'Spearman'); %correlation between true and predicted NF score
xlabel('True NF score');
ylabel('Predicted NF score');
title(sprintf('Validation: rho = %.3f (p = %.3f)', rho, pval));


%AUC
C_test = Y_test > median(Y_test);
figure;
%clf(fig1);
[fpr,tpr,~,AUC] = perfcurve(C_test, Y_pred, 1);
plot(fpr,tpr,'Color',co(2,:));
hold on;
plot([0,1],[0,1],'k:');
axis square;
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title(sprintf('ROC Curve for validation (AUC = %.3f)',AUC));


