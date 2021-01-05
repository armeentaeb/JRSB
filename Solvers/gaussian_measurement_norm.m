function [rank,low_rank_estimate] = gaussian_measurement_norm(y,A,lambda)

p = sqrt(size(A,2));
cvx_precision medium
cvx_begin 
variable L(p,p) symmetric
minimize 1/length(y)*sum_square(y-A*reshape(L,p^2,1)) + lambda*norm_nuc(L)
cvx_end

rank = length(find(svd(L) > 10^(-3)));
[U D V] = svd(L);
output_proj_U = U(:,1:rank)*U(:,1:rank)';
output_proj_V = V(:,1:rank)*V(:,1:rank)';
A = L;
low_rank_estimate = L;
