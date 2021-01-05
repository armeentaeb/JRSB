function [L,rk] = gaussian_measurement_norm_large_scale(y,A,k,p1,p2,lambda,Uinit,Vinit)
 

% 
% U = randn(p1,k);
% U = U/norm(U);
% V = randn(p2,k)';
% V = V/norm(V);
% learn_param = 0.5;
% L = U*V;
% Lnew = randn(p1,p2);
% while norm(L-Lnew) >= 10^(-3)
%     L = Lnew;
%     grad = zeros(p1,p2);
%    for i = 1:length(y)
%      grad = reshape(A(i,:),p1,p2)'*trace(reshape(A(i,:),p1,p2)*Lnew)-...
%             reshape(A(i,:),p1,p2)'*y(i)+grad;
%    end
%     L = L-learn_param*grad/length(y);
%     [U D V] = svd(L);
%     Lnew = U(:,1:k)*D(1:k,1:k)*V(:,1:k)';  
%      norm(L-Lnew)
% end
% 
% % 
% % 
% % 
% % 
% % 
% % 
% 
% % method 2
% U = randn(p1,k);
% U = U/norm(U);
% V = randn(p2,k)';
% V = V/norm(V);
% Uold = randn(p1,k);
% Vold = randn(p2,k)';
% 
% mat = A'*A;
% dat = A'*y;
% learn_param = 0.05;
% while norm(U*V-Uold*Vold,'fro')/norm(Uold*Vold,'fro') > 10^(-2)
%     Uold = U; Vold = V;
%     Z1 = (kron(eye(p2),U));
%     Vold = V;
%     V = Vold-learn_param/length(y)*reshape((Z1'*mat*Z1)*reshape(Vold,k*p2,1)-Z1'*dat,k,p2);   
%        V = reshape((Z1'*mat*Z1+lambda*eye(size(Z1,2)))^(-1)*Z1'*dat,k,p2);
%     Z2 = (kron(V',eye(p1)));
%     U = Uold-learn_param/length(y)*reshape(Z2'*mat*Z2*reshape(Uold,p1*k,1)-Z2'*dat,p1,k);
%     U = reshape((Z2'*mat*Z2+lambda*eye(size(Z2,2)))^(-1)*Z2'*dat,p1,k);
%     norm(U*V-Uold*Vold,'fro')/norm(Uold*Vold,'fro')
% end
% L = U*V;
% rk = size(U,2);
% 
% 
% 
% 
% 
% 
% 
% 
% 

%% method 3
U = Uinit;
U = U/norm(U);
V = Vinit';
V = V/norm(V);
Uold = randn(p1,k);
Vold = randn(p2,k)';

mat = A'*A;
dat = A'*y;

while max(max(abs(U*V-Uold*Vold)))/max(max(abs(Uold*Vold))) > 10^(-3)
    Uold = U; Vold = V;
    Z1 = (kron(eye(p2),U));
    V = reshape((Z1'*mat*Z1+lambda*eye(size(Z1,2)))^(-1)*Z1'*dat,k,p2);
    Z2 = (kron(V',eye(p1)));
    U = reshape((Z2'*mat*Z2+lambda*eye(size(Z2,2)))^(-1)*Z2'*dat,p1,k);
    max(max(abs(U*V-Uold*Vold)))/max(max(abs(Uold*Vold)));
end;
L = U*V;
rk = size(U,2);
%      
%      
%      
%     