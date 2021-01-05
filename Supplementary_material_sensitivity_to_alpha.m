%% Written by Armeen Taeb, California Institute of Technology, Oct 2018
%% This is a code generates quantities for Figure 5b in the linear measurement setting
%% In particular, `vanilla_FD' number of false discoveries
%% for top 10 ranks with the no-subsampling procedure and `WS_top_projection'
%% finds the number of false discoveries for top 10 ranks with WS-subsampling
%% procedure
clear all; close all; clc;
root = pwd;
addpath(strcat(root,'/Solvers'));
addpath(strcat(root,'/Solvers/cvx'));

p = 100;
num_bags = 50;
SNR = [0.5,0.8,2];
rank_vector = [1,3,5];
alpha_vec = linspace(0.6,0.8,9);
lam_vector = linspace(0.01,1.2,100);
num_iter = 100;
rank_const = 10;

total_configs = length(rank_vector)*length(SNR)*length(alpha_vec);
NS_FD = zeros(num_iter,total_configs);
WS_FD = zeros(num_iter,total_configs);
NS_power = zeros(num_iter,total_configs);
WS_power = zeros(num_iter,total_configs);
incoherence_val = zeros(num_iter,total_configs);

t0 = 0.3;
t_train = 0.3;
t_test = 0.1;
config = 0;
for rank_ind = 1:length(rank_vector)
    
    rank = rank_vector(rank_ind); rank_star = rank;
    rankl = floor(rank*1/2);
    
    [U D V] = svd(randn(p));
    Us = U;
    Vs = V;
    Lstar = zeros(p);
    for i = 1:rankl
        sigma = 1;
        Lstar = Lstar + sigma*Us(:,i)*Vs(:,i)';
    end
    
    for i = rankl+1:rank
        sigma =   1;
        Lstar = Lstar + sigma*Us(:,i)*Vs(:,i)';
    end
    
    
    perp_dirs_U = Us(:,rank+1:end)*Us(:,rank+1:end)';
    perp_dirs_V = Vs(:,rank+1:end)*Vs(:,rank+1:end)';
    
    for noise_ind = 1:length(SNR)
        
        snr = SNR(noise_ind);
        noise_ind
            
        % fix training and test dataset
        Data_Obs = zeros(p);
        t = randperm(p^2);
            Data_Obs(t(1:0.6*p^2)) = 1;
            
            
        Data_not_Obs = Data_Obs == 0;
        ind = find(Data_not_Obs == 1);
        t = randperm(length(ind));
        Data_test = Data_not_Obs;
        Data_test(ind(t(1:0.5*length(t)))) = 0;
            
           
       for j = 1:length(alpha_vec)
            alpha_thresh = alpha_vec(j);
            config = config+1;
            
            
             for iter = 1:num_iter
                
                % find the noise level to get desired SNR
                temp_vec = linspace(0.001,1,1000);
                for i = 1:length(temp_vec)
                    Noise = temp_vec(i)/sqrt(p)*randn(p);
                    if norm(Lstar,'fro')/norm(Noise,'fro') < snr
                        break;
                    end
                end
                Data = (Lstar + Noise);
                
                
                validation = 100;
                for i = 1:length(lam_vector)
                    
                    lambda = lam_vector(i);
                    %[rank_regular(i),~,~,A,low_rank_estimate] = matrix_completion_nuclear_norm(Data',Data_Obs',size(Lstar,1),lambda);
                    %[Lnew] = matrix_completion_large_scale(Data,Data_Obs,2,lambda);
                    [rank,output_proj_U,output_proj_V,A,low_rank_estimate] = matrix_completion_ALS(Data',Data_Obs',rank_const,lambda);

                    
                    [U,~,V] = svd(low_rank_estimate);
                    validation_performance(i) = norm((low_rank_estimate-Data).*Data_test,'fro');
                    if validation_performance < validation
                        validation = validation_performance;
                    else
                        break;
                    end
                end
                lambda = lam_vector(i-1);
                
                
                % compute FD and PW without subsampling
                [rank_reg(config,iter),~,~,A,low_rank_estimate] = matrix_completion_ALS(Data',Data_Obs',rank_const,lambda);
                [U,~,V] = svd(low_rank_estimate);
                temp = ((p-rank_star)^2 - ...
                    trace(U(:,rank_reg(config,iter)+1:end)*U(:,rank_reg(config,iter)+1:end)'*perp_dirs_U)* ...
                    trace(V(:,rank_reg(config,iter)+1:end)*V(:,rank_reg(config,iter)+1:end)'*perp_dirs_V));
                
                
                NS_FD(iter,config) = NS_FD(iter,config) + temp/(p-rank_star)^2;
                NS_power(iter,config) = NS_power(iter,config) + (2*p*rank_reg(config,iter)-rank_reg(config,iter)^2-temp)/(2*p*rank_star-rank_star^2);
                
                
                
                
                % compute FD and PW with subsampling
                avg_proj_matrix_U  = zeros(p);
                avg_proj_matrix_V  = zeros(p);
                avg_tangent_space = zeros(p^2);
                row_bags = cell(num_bags*2,1);
                col_bags = cell(num_bags*2,1);
                
                for j = 1:num_bags
                    
                    data_subsampled_1 = Data_Obs;
                    ind = find(Data_Obs == 1);
                    t = randperm(length(ind));
                    data_subsampled_1(ind(t(floor(length(t)/2)+1:end))) = 0;
                    [rank,output_proj_stab_U,output_proj_stab_V,A,low_rank_estimate] = matrix_completion_ALS(Data',Data_Obs',rank_const,lambda);     
                    avg_proj_matrix_U = avg_proj_matrix_U  + output_proj_stab_U;
                    avg_proj_matrix_V  = avg_proj_matrix_V  + output_proj_stab_V;
                    row_bags{2*j-1} = output_proj_stab_V;
                    col_bags{2*j-1} = output_proj_stab_U;
                    
                    
                    data_subsampled_2 = Data_Obs;
                    data_subsampled_2(ind(t(1:floor(length(t)/2)))) = 0;
                    [~,output_proj_stab_U,output_proj_stab_V,~,~] = matrix_completion_ALS(Data',data_subsampled_2',rank_const,lambda);
                    avg_proj_matrix_U = avg_proj_matrix_U  + output_proj_stab_U;
                    avg_proj_matrix_V = avg_proj_matrix_V  + output_proj_stab_V;
                    row_bags{2*j} = output_proj_stab_V;
                    col_bags{2*j} = output_proj_stab_U;
                end
                
                
                
                
                [U, D_U ,~] = svd(avg_proj_matrix_U/(num_bags*2));
                [V, D_V ,~] = svd(avg_proj_matrix_V/(num_bags*2));
                
               % P_avg = zeros(p^2);
               % for j = 1:num_bags
                %    P_avg  =  P_avg  + kron((eye(p)-col_bags{2*j-1}),(eye(p)-row_bags{2*j-1}));
                %    P_avg  =  P_avg  + kron((eye(p)-col_bags{2*j}),(eye(p)-row_bags{2*j}));
                %end
                
                
                % find largest rank satisfying stability selection
                % criterion
               % k = 0;
               % alpha_val = 1;
               % while alpha_val > alpha_thresh
                %    k = k+1;
                %    P_T = eye(p^2) - kron(U(:,k+1:end)*U(:,k+1:end)',V(:,k+1:end)*V(:,k+1:end)');
                %    alpha_val = 1-norm(P_T*P_avg/(2*num_bags)*P_T);
                %    alpha(k,noise_ind) = alpha_val;
                %end
                
                k = min(length(find(diag(D_U)>alpha_thresh)),length(find(diag(D_V)>alpha_thresh)));
                rank_stab(config,iter) = k;
                
                % computing number of false discoveries for subspace
                % stability selection
                temp = ((p-rank_star)^2 - ...
                    trace(U(:,rank_stab(config,iter)+1:end)*U(:,rank_stab(config,iter)+1:end)'*perp_dirs_U)* ...
                    trace(V(:,rank_stab(config,iter)+1:end)*V(:,rank_stab(config,iter)+1:end)'*perp_dirs_V));
                
                WS_FD(iter,config) = WS_FD(iter,config) + temp/(p-rank_star)^2;
                WS_power(iter,config) = WS_power(iter,config) + (2*p*rank_stab(config,iter)-rank_stab(config,iter)^2-temp)/(2*p*rank_star-rank_star^2);
                
                
                
            end
            mean_WS_FD(config) = mean(WS_FD(:,config));
            mean_WS_power(config) = mean(WS_power(:,config));
            mean_NS_power(config) = mean(NS_power(:,config));
            mean_NS_FD(config) = mean(NS_FD(:,config));
 
            var_WS_FD(config) = var(WS_FD(:,config));
            var_WS_power(config) = var(WS_power(:,config));
            var_NS_power(config) = var(NS_power(:,config));
            var_NS_FD(config) = var(NS_FD(:,config));
 
            
        end
    end
end



