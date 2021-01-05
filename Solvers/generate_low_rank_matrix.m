function [Lstar] = generate_low_rank_matrix(dynamic_sing_vals,p,inc,rank)

% U = eye(p);
% V = eye(p);
% ins_s = 0;
% temp_vec = linspace(10,1000,1000);
% for i =1:length(temp_vec)
%     [U1,~,~] = svd(U(:,1:rank)*U(:,1:rank)' + randn(p)/temp_vec(i));
%     [V1,~,~] = svd(V(:,1:rank)*V(:,1:rank)' + randn(p)/temp_vec(i));    % setup population model
%     if max(diag(U1(:,1:rank)*U1(:,1:rank)')) >= inc
%         U = U1;V = V1;
%         break;
%     end
% end
% Lstar = zeros(p);


[U,~,V] = svd(randn(p,p));

i = 1;
vec = [];
while i <= rank
    if i >= 1 & i <= rank
        vec = [vec;dynamic_sing_vals(1)];
    end
    i = i+1;
end

vec = [100;70;30;30];
Lstar = U(:,1:rank)*diag(vec)*U(:,1:rank)';


end