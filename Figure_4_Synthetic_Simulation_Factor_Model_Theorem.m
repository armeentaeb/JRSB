%% Written by Armeen Taeb, California Institute of Technology, December 2016
%% This is a code that generates the quantities for Figure 4
%% In particular, `false_discovery_stability' represents stability selection
%% false discovery and `false_discovery_bound' represents theorem bound

clear all
clc
pwd
root = pwd;
addpath(strcat(root,'/Solvers'));


%% setup model parameters
num_bags = 25;
p = 200;
lam_vector = [linspace(0.15,1,50)];
alpha_vec = linspace(0.5,0.99,100); snr_vec = [0.8]; n_vec = [150*p]; rk_vec = [1,10]; inc_vec = [0.4];
num_iter = 50;

false_discovery_stability_7 = zeros(length(alpha_vec),length(snr_vec)*length(n_vec)*length(rk_vec)*length(inc_vec));
power_stability_7 = zeros(length(alpha_vec),length(snr_vec)*length(n_vec)*length(rk_vec)*length(inc_vec));


config = 0;
for n_ind = 1:length(n_vec)
    for rank_ind = 1:length(rk_vec)
        for snr_ind = 1:length(snr_vec)
            for inc_ind = 1:length(inc_vec)
                
                
                
                
                config = config+1;
                sprintf('this is config %d\n', config)
                
                rank_star = rk_vec(rank_ind);
                n = n_vec(n_ind);
                snr = snr_vec(snr_ind);
                inc = inc_vec(inc_ind);
                
                
                %% setup model
                [U,~,~] = svd(eye(p));
                temp = linspace(5,1000,10000);
                for i = 1:length(temp)
                    [Us,~,~] = svd(U(:,1:rank_star)*U(:,1:rank_star)'+randn(p,p)/temp(i));
                    if max(diag(Us(:,1:rank_star)*Us(:,1:rank_star)'))>inc
                        break;
                    end
                end
                
                
                D = eye(p);
                vec = []; j = 1;
                while j <= rank_star
                    if j == 1
                        vec = [vec;snr];
                    elseif j <= 3 & i >= 2
                        vec = [vec;snr/2];
                    else
                        vec = [vec;snr/4];
                    end
                    j = j+1;
                end
                Lstar =  0.8*Us(:,1:rank_star)*Us(:,1:rank_star)';
                temp = randn(p)/(25*sqrt(2));
                [U D V] = svd((diag(svd(Lstar)))+temp*temp');
                Lstar = U(:,1:rank_star)*D(1:rank_star,1:rank_star)*U(:,1:rank_star)';
                D = eye(p);
                Sigma = (6*D-6*Lstar)^(-1);
                
                %% population tangent space
                [Ustar,~,~] = svd(Lstar);
                perp_dirs_U = Ustar(:,rank_star+1:end)*Ustar(:,rank_star+1:end)';
                P_Tstar_perp = Ustar(:,rank_star+1:end)*Ustar(:,rank_star+1:end)';
                
                
                
                selected_dim = zeros(length(alpha_vec),1);
                temp_full =  0;
                temp_half_data_trace = 0;
                temp_half_data_trace_variance = 0;
                temp_data_interactions = zeros(length(alpha_vec),1);
                temp_half_data_basis_dependent = zeros((p-rank_star)^2,1);
                temp_half_data_basis_dependent_nuc = zeros((p-rank_star)^2,1);
                nu = 0;
                Data = mvnrnd(zeros(p,1),Sigma,n);
                Y = Data;
                tr_tt_perm = randperm(size(Data,1));
                
                % setup training and testing data
                Y_train = Y(tr_tt_perm(1:floor(length(tr_tt_perm)*70/100)),:);
                Y_test = Y(tr_tt_perm(floor(length(tr_tt_perm)*70/100)+1:length(tr_tt_perm)),:);
                
                
                t = randperm(size(Y_train,1));
                Y_train_sub = Y_train(t(1:length(t)/2),:);
                for lam_ind = 1:length(lam_vector)
                    
                    lambda = lam_vector(lam_ind);
                    [F,rank_regular(lam_ind),valid(lam_ind)] = factor_model(Y_train_sub,Y_test,lambda);
                    Sigma_test = diag(diag(cov(Y_train)).^(-1/2))*cov(Y_test)* diag(diag(cov(Y_train)).^(-1/2));
                    validation(lam_ind)= -log(det(F{1}))+trace(F{1}*Sigma_test);
                    %
                    
                    rank_regular(lam_ind) = length(find(svd(F{2})>10^(-2)));
                    proj_matrix = U(:,1:rank_regular(lam_ind))*U(:,1:rank_regular(lam_ind))';
                    if rank_regular(lam_ind) == 0
                        break;
                    end
                end
                
                
                
                [~,ind] = min(validation); lambda = lam_vector(ind);
                
                
                %% compute empirical expectation over many trials
                for iter = 1:num_iter
                    sprintf('this is iteration %d\n', iter)
                    
                    % generate data
                    Data = mvnrnd(zeros(p,1),Sigma,n);
                    Y = Data;
                    tr_tt_perm = randperm(size(Data,1));
                    
                    % setup training and testing data
                    Y_train = Y(tr_tt_perm(1:floor(length(tr_tt_perm)*70/100)),:);
                    Y_test = Y(tr_tt_perm(floor(length(tr_tt_perm)*70/100)+1:length(tr_tt_perm)),:);
                    
                    %% sweep over lambda to choose cross-validated choice
                    for lam_ind = 1:length(lam_vector)
                        
                        lambda = lam_vector(lam_ind);
                        [F,rank_regular(lam_ind),valid(lam_ind)] = factor_model(Y_train,Y_test,lambda);
                        Sigma_test = diag(diag(cov(Y_train)).^(-1/2))*cov(Y_test)* diag(diag(cov(Y_train)).^(-1/2));
                        validation(lam_ind)= -log(det(F{1}))+trace(F{1}*Sigma_test);
                        %
                        %
                        rank_regular(lam_ind) = length(find(svd(F{2})>10^(-2)));
                        %                         proj_matrix = U(:,1:rank_regular(lam_ind))*U(:,1:rank_regular(lam_ind))';
                        %                         if rank_regular(lam_ind) == 0
                        %                             break;
                        %                         end
                    end
                    
                    %% use full data and compute false discovery
                    [F,rank_regular(lam_ind),valid(lam_ind)] = factor_model(Y_train,Y_test,lambda);
                    [U,dp,~] = svd(F{2}); rk = length(find(dp>10^(-2)));
                    temp_full = temp_full + trace(U(:,1:rk)*U(:,1:rk)'*perp_dirs_U)*trace(perp_dirs_U);
                    
                    %% use half data and compute false discovery
                    t = randperm(size(Y_train,1));
                    Y_train_sub = Y_train(t(1:length(t)/2),:);
                    [F,~,~] = factor_model(Y_train_sub,Y_test,lambda);
                    [U,dp,~] = svd(F{2});
                    rk = length(find(dp>10^(-2)));
                    temp_half_data_trace = temp_half_data_trace+ trace(U(:,1:rk)*U(:,1:rk)'*perp_dirs_U);
                    P_That = U(:,1:rk)*U(:,1:rk)';
                    
                    %% compute basis dependent results
                    k = 1;
                    for i = 1:p-rank_star
                        basis_Tperp(1:p,k) = Ustar(:,rank_star+j);
                        k = k+1;
                    end
                    
                    
                    for i = 1:(p-rank_star)
                        temp_half_data_basis_dependent(i) = temp_half_data_basis_dependent(i)+norm(P_That*basis_Tperp(:,i));
                    end
                    
                    
                    
                    
                    %% bagging to see what stability selection would yield
                    %                     avg_proj_matrix_U = zeros(size(Data,2));
                    %                     avg_proj_matrix_V = zeros(size(Data,1));P_avg = zeros(p^2);
                    %
                    %                     for bag_counter = 1:num_bags
                    %                         t = randperm(size(Y_train,1));
                    %                         Y_train_sub = Y_train(t(1:length(t)/2),:);
                    %                         [F,~,~] = factor_model(Y_train_sub,Y_test,lambda);
                    %                         [U,dp,~] = svd(F{2}); rk = length(find(dp>10^(-2)));
                    %                         output_proj_stab_U = U(:,1:rk)*U(:,1:rk)';
                    %                         avg_proj_matrix_U = avg_proj_matrix_U  + output_proj_stab_U;
                    %
                    %                         % compute average projection matrix
                    %                         P_avg = P_avg +  kron(eye(p)-output_proj_stab_U,eye(p)-output_proj_stab_U);
                    %
                    %
                    %                         Y_train_sub = Y_train(t(length(t)/2+1:end),:);
                    %                         [F,~,~] = factor_model(Y_train_sub,Y_test,lambda);
                    %                         [U,dp,~] = svd(F{2});
                    %                         rk = length(find(dp>10^(-2)));
                    %                         output_proj_stab_U = U(:,1:rk)*U(:,1:rk)';
                    %                         avg_proj_matrix_U = avg_proj_matrix_U  + output_proj_stab_U;
                    %
                    %                         % compute average projection matrix
                    %                         P_avg = P_avg +  kron(eye(p)-output_proj_stab_U,eye(p)-output_proj_stab_U);
                    %
                    %                     end
                    %
                    %
                    %                     %[U,D,V] = svd(avg_proj_matrix_U/(2*num_bags));
                    %                     P_avg = P_avg/(2*num_bags);
                    %                     %tot_dim = tot_dim+(p^2-trace(P_avg));
                    %                     %alpha_val = diag(D(1:rank_star+1,1:rank_star+1));
                    %
                    %                     alpha_val = zeros(rank_star+1,1);
                    %                     for k = 1:rank_star+1
                    %                         P_T = eye(p^2) - kron(U(:,k+1:end)*U(:,k+1:end)',U(:,k+1:end)*U(:,k+1:end)');
                    %                         alpha_val(k) = 1-norm(P_T*P_avg*P_T);
                    %                     end
                    %
                    %                     for j = 1:length(alpha_vec)
                    %
                    %                         alpha_thresh = alpha_vec(j);
                    %                         k = length(find(alpha_val >= alpha_thresh));
                    %                         proj_matrix = U(:,k+1:end)*U(:,k+1:end)';
                    %                         temp = ((p-rank_star)^2 - trace(proj_matrix*perp_dirs_U)* trace(proj_matrix*perp_dirs_U));
                    %                         false_discovery_stability_7(j,config) =false_discovery_stability_7(j,config) + temp;
                    %                         power_stability_7(j,config) = power_stability_7(j,config) + (2*p*k-k^2-temp)/(2*p*k-k^2);
                    %                         sing = svd(avg_proj_matrix_U/(2*num_bags)); opt_tang_rk = 2*p*k-k^2;
                    %                         selected_dim(j) = selected_dim(j)+opt_tang_rk;
                    %                     end
                    %                 end
                    %
                    nu = norm(P_That*P_Tstar_perp-P_Tstar_perp*P_That,'fro')+ nu;
                    k = 1;
        for i = 1:p-rank_star
            basis_Tperp(1:p,k) = Ustar(:,rank_star+i);
            k = k+1;
        end
        
        
       for i = 1:p-rank_star
            commutator(i) = commutator(i)+norm(P_That*Ustar(:,rank_star+i)*Ustar(:,rank_star+i)'-Ustar(:,rank_star+i)*Ustar(:,rank_star+i)'*P_That,'fro');
        end
                end
                Term_full_nostab = temp_full/iter;
                Term_stability_nobasis = (temp_half_data_trace_variance/iter)^2;
                Term_half_nostab = temp_half_data_trace/iter;
                Term_basis = sum((temp_half_data_basis_dependent/iter).^2);
                Term_basis_fourth = sum((temp_half_data_basis_dependent_nuc/iter).^2);
                Term_interactions = temp_data_interactions/iter;
                
                
                
                avg_selected_dim(1:length(alpha_vec),config) = selected_dim/iter;
                quality_bag(config,1:4) = [Term_full_nostab, Term_half_nostab, ...
                    Term_stability_nobasis, Term_basis];
            end
        end
    end
end













% compute stability selection false discovery

% compare true false discovery/bound/dim(T^\star)^\perp/previous bound
false_discovery_stability_7 = false_discovery_stability_7/50;

figure;
plot(alpha_vec,quality_bag(1,4)+4*sqrt(1-alpha_vec)'.*avg_selected_dim(:,1))
hold on;
plot(alpha_vec,false_discovery_stability_7(:,1))
hold on;
plot(alpha_vec,quality_bag(1,4)+4*sqrt(1-alpha_vec)'.*(p-6)^2)
hold on;
plot(alpha_vec,(p-6)^2*ones(1,length(alpha_vec)),'--')

figure;
plot(alpha_vec,quality_bag(1,4)+4*sqrt(1-alpha_vec)'.*avg_selected_dim(:,1))
hold on;
plot(alpha_vec,false_discovery_stability_7(:,1))



% see how much is due to quality of estimator and how much role of alpha
figure;
plot(quality_bag(2,5)+4*sqrt(1-alpha_vec)'.*(p-10)^2)
hold on;plot(quality_bag(2,5))
hold on;plot(4*sqrt(1-alpha_vec)'.*avg_selected_dim(:,2))

