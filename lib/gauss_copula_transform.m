function [udata, out_copula]= gauss_copula_transform(data, in_copula)

udata = zeros(size(data));
for dim=1:size(data,2),
  if nargin<2||isempty(in_copula),
    [ecdf_ x]= ecdf(data(:,dim));

    %the precision of the extreme value is half way between the last two
    %values
    myeps = (ecdf_(end) - ecdf_(end-1))/2;
    ecdf_(end)=1-myeps;
    ecdf_(1)=0+myeps;

    %find duplicates and add random noise to break it
    [xu, tmp, index] = unique(x);
    dup_index = find(diff(index)==0);
    x(dup_index) = x(dup_index) + rand * myeps;
  else %use already derived copula transforms
    x = in_copula{dim}.x;
    ecdf_ = in_copula{dim}.ecdf;
  end;
  f = interp1(x,ecdf_,data(:,dim),'linear');
  %   cla; hold on;
  %   plot(x, ecdf_,'b-');
  %   plot(data(:,dim), f,'ro');
  udata(:,dim) = norminv(f);
  out_copula{dim}.x = x;
  out_copula{dim}.ecdf = ecdf_;
end;
