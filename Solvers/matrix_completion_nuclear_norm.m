% This function takes in data with subset of indices that are observed and
% a regularization parameter lambda and uses the nuclear norm matrix
% completion estimator to find a low rank matrix

function [rank,output_proj_U,output_proj_V,A,low_rank_estimate] = matrix_completion_nuclear_norm(Data,Obs,k,lambda)

Data = Data';
Obs = Obs';
U = randn(size(Data,1),k);
U = U/norm(U);
V = randn(size(Data,2),k);
V = V/norm(V);

cvx_begin quiet
variables L(size(Data))
minimize square_pos(norm((Data-L).*Obs,'fro')) + lambda*norm_nuc(L)
cvx_end

rank = length(find(svd(L) > 10^(-3)));
[U D V] = svd(L);
output_proj_U = U(:,1:rank)*U(:,1:rank)';
output_proj_V = V(:,1:rank)*V(:,1:rank)';
A = L;
low_rank_estimate = L;
