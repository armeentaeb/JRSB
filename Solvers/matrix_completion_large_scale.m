function [Lnew] = matrix_completion_large_scale(Data,Obs,k,lambda)

Lold = randn(size(Data));
Lnew = randn(size(Data));

Gamma_old = randn(size(Data));
Gamma_old(find(Obs == 1)) = 0;

Gamma_new = randn(size(Data));
Gamma_new(find(Obs == 1)) = 0;


Data(find(Obs == 0)) = 0;

while max(norm(Lold-Lnew)/norm(Lold),norm(Gamma_old-Gamma_new)/norm(Gamma_old))> 10^(-4)
    
    Lold = Lnew;
    Gamma_old = Gamma_new;
    [U,D,V] = svd(Data+Gamma_new);
    D(D < 2*lambda) = 0;
    Lnew = U*D*V'; 
    Gamma_new(find(Obs == 0)) = Lnew(find(Obs == 0));
end
       
    
end    
