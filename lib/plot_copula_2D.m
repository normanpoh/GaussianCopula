function [llhmat, param, xtesta1, xtesta2] = plot_copula_2D(data, axis_)

if nargin <3,
  shadinglevel=10;
end;

%data is 2D

[udata, param.copula] = gauss_copula_transform(data);
param.mean = mean(udata);
param.cov = cov(udata);

[xtesta1,xtesta2]=meshgrid( ...
  linspace(axis_(1), axis_(2), 100), ...
  linspace(axis_(3), axis_(4), 100) );
[na,nb]=size(xtesta1);
xtest1=reshape(xtesta1,1,na*nb);
xtest2=reshape(xtesta2,1,na*nb);
xtest=[xtest1;xtest2]';

%%
llh = mvnpdf(gauss_copula_transform(xtest, param.copula),param.mean,param.cov);
%%
llh(isnan(llh))=realmin;
llhmat=reshape(llh,na,nb);
%contourf(xtesta1,xtesta2,llhmat,shadinglevel);shading flat;
%[~,h]=contour(xtesta1,xtesta2,ypredmat,5); %shading flat;
%set(h,'ShowText','on','TextStep',get(h,'LevelStep')*2)

