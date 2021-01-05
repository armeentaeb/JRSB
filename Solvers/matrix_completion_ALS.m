function [rank,output_proj_U,output_proj_V,A,low_rank_estimate] = matrix_completion_ALS(Data,Obs,k,lambda)

Data = Data';
Obs = Obs';
U = randn(size(Data,1),k);
U = U/norm(U);
V = randn(size(Data,1),k);
V = V/norm(V);
Uold = randn(size(Data,1),k);
Vold = randn(size(Data,2),k);


while norm(U*V'-Uold*Vold','fro')^2/norm(Uold*Vold','fro')^2 > 5*10^(-3)

    Uold = U;
    Vold = V;
     %U = Data*V*(V'*V)^(-1);
     
    for col = 1:size(Data,1)
        ind = find(Obs(col,:) == 1); 
        if length(ind)~=0
           U(col,:) = ((V(ind,:)'*V(ind,:)+lambda*eye(k))^(-1)*(V(ind,:)'*Data(col,ind)'));
        end
    end
    for row = 1:size(Data,2)
        ind = find(Obs(:,row) == 1); 
        if length(ind)~=0
             V(row,:) = ((U(ind,:)'*U(ind,:)+lambda*eye(k))^(-1)*(U(ind,:)'*Data(ind,row)))';
        end
    end
    
      error = norm(U*V'-Uold*Vold','fro')/norm(Uold*Vold','fro');
end    

rank = length(find(svd(U*V') > 10^(-3)));
[A,~,~] = svd(U);
A = A(:,1:rank);
[B,~,~] = svd(V);
output_proj_U = A(:,1:rank)*A(:,1:rank)';
output_proj_V = B(:,1:rank)*B(:,1:rank)';
low_rank_estimate = U*V';

