clear
close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
FC_all=rand(65, 253)*2-1;
NFscore_all=rand(65, 1);
mask_train=[true(41,1); false(24,1)];

X_train = FC_all(mask_train,:);
Y_train = NFscore_all(mask_train);
N_train = length(Y_train);
X_test = FC_all(~mask_train,:);
Y_test = NFscore_all(~mask_train);

var_idx = cell(10000000,1);
idx = 0;
for p = 1:28
    tmp = combnk(1:28,p);
    for k = 1:size(tmp,1)
        idx = idx+1;
        var_idx{idx} = tmp(k,:);
    end
end
var_idx(idx+1:end) = [];
MaxK = length(var_idx);
Fitness = nan(1,MaxK);
for k = 1:MaxK
	if mod(k, 1000000)==0
	disp([k,MaxK])
	end
    mask_var = false(1,28);
    mask_var(var_idx{k}) = true;
    Y_pred = nan(size(Y_train));
    for n = 1:N_train
        x1 = X_train(:,mask_var);
        x1(n,:) = [];
        y1 = Y_train;
        y1(n) = [];
        [~,~,~,~,betaPLS] = plsregress(x1,y1,1);
        x0 = X_train(n,mask_var);
        Y_pred(n) = [ones(size(x0,1),1), x0]*betaPLS;
    end
    [Fitness(k),~] = corr(Y_train, Y_pred, 'type', 'Spearman');
end
%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5

[~,max_index]=max(Fitness);
%[~,sort_Fitness]=sort(Fitness,'descend');
%for i_search=1:1000
%max_index=sort_Fitness(i_search);
var_best = var_idx{max_index};
for n = 1:N_train
    x1 = X_train(:,var_best);
    x1(n,:) = [];
    y1 = Y_train;
    y1(n) = [];
    [~,~,~,~,betaPLS] = plsregress(x1,y1,1);
    x0 = X_train(n,var_best);
    Y_pred(n) = [ones(size(x0,1),1), x0]*betaPLS;
end
figure;
%clf(fig1);
scatter(Y_train, Y_pred, 100,co(1,:),'filled');

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


%
%
%
[Xloadings,Yloadings,Xscores,Yscores,betaPLS] = plsregress(X_train(:,var_best),Y_train,1);
Y_pred = [ones(size(X_test(:,var_best) ,1),1) X_test(:,var_best)]*betaPLS;
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
[rho, pval] = corr(Y_test, Y_pred, 'type', 'Spearman');
xlabel('True NF score');
ylabel('Predicted NF score');
title(sprintf('Validation: rho = %.3f (p = %.3f)', rho, pval));



% X_test2 = FC_all(mask_mdd,var_best);
% Y_test2 = NFscore_all(mask_mdd);
% Y_pred2 = [ones(size(X_test2 ,1),1) X_test2]*betaPLS;
% [rho2, pval2] = corr(Y_test2, Y_pred2, 'type', 'Spearman')
% scatter(Y_test2, Y_pred2, 100,co(1,:),'filled');

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

%end

