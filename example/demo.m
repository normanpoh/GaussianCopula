%%
addpath ../lib

d=1; 

%% load impostor data in NIST format
[expe.dset{d,1}, tmp, expe.label{d,1}, tmp] = load_raw_scores_labels('imp.scores');
[tmp, expe.dset{d,2}, tmp, expe.label{d,2}] = load_raw_scores_labels('gen.scores');
clear tmp

%% let's go
expe.dset{2,1} = expe.dset{1,1};
expe.dset{2,2} = expe.dset{1,2};
expe.label{2,1} = expe.label{1,1};
expe.label{2,2} = expe.label{1,2};
%%
[com, epc_cost] = fusion_gauss_copula(expe, [1 2],[],[], 1)

%%


%%
clf; hold on;
scatter(expe.dset{1,1}(:,1),expe.dset{1,1}(:,2));
scatter(expe.dset{1,2}(:,1),expe.dset{1,2}(:,2));
axis_ = axis;
%% get the likelihood maps
[llhmat, param, xtesta1, xtesta2]  = plot_copula_2D(expe.dset{1,1},axis_);
[llhmat2, param, xtesta1, xtesta2]  = plot_copula_2D(expe.dset{1,2},axis_);
%%
clf;
contourf(xtesta1,xtesta2,llhmat + llhmat2,10);shading flat;
hold on;
scatter(expe.dset{1,1}(1:200,1),expe.dset{1,1}(1:200,2));
scatter(expe.dset{1,2}(1:200,1),expe.dset{1,2}(1:200,2));



