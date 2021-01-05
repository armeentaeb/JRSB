function [D,L] = large_scale_factor_analysis(Data)

Sigma = cov(Data);
p = size(Data,1);
diagTerm = rand(p);

while (Lold-Lnew)
    
    [U D V] = svd(Sigma-diagTerm);
    D(find(D < lambda)) = 0;
    L = U*D*V';
    
    D = Sigma-L
