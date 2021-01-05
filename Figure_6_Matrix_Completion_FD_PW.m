%% Written by Armeen Taeb, California Institute of Technology, Oct 2018
%% This is a code generates quantities for Figure 5a in the matrix completion setting
%% In particular, `NS_FD', 'NS-power': number of false discoveries and power for the no subsampling approach
%% `WS_FD', 'WS-power': number of false discoveries and power for the subsampling approach
clc;close all;clear all;
root = pwd;
addpath(strcat(root,'/Solvers'));
addpath(strcat(root,'/Solvers/cvx'));

p = 100;
num_bags = 50;
SNR = linspace(0.5,2,5);
rank_vector = [1,2,3,4];
alpha_thresh = 0.7;
num_iter = 100;
rank_const = 10;
total_configs = length(rank_vector)*length(SNR);
NS_FD = zeros(num_iter,total_configs);
WS_FD = zeros(num_iter,total_configs);
NS_power = zeros(num_iter,total_configs);
WS_power = zeros(num_iter,total_configs);
incoherence_val = zeros(num_iter,total_configs);


lam_vector = linspace(0,0.2,100);
noise_vector = linspace(0.1,0.5,10);
t0 = 0.3;
k = 10;

config = 0;
for rank_ind = 1:length(rank_vector)
    
    rank_star = rank_vector(rank_ind);
    rankl = floor(rank_star*1/2);
    
    % generate the low rank matrix
    [U,~,~] = svd(randn(p));
    Lstar = zeros(p);
    for i = 1:rankl
        sigma = 1;
        Lstar = Lstar + sigma*U(:,i)*U(:,i)';
    end
    
    for i = rankl+1:rank_star
        sigma =   1;
        Lstar = Lstar + sigma*U(:,i)*U(:,i)';
    end
    [U,~,V] = svd(Lstar);
    perp_dirs_U = U(:,rank_star+1:end)*U(:,rank_star+1:end)';
    perp_dirs_V = V(:,rank_star+1:end)*V(:,rank_star+1:end)';
    Us = U;
    
    
    for noise_ind = 1:length(SNR)
        noise_ind
        config = config+1;
        
        % fix training and test dataset
        Data_Obs = zeros(p);
        t = randperm(p^2);
        Data_Obs(t(1:0.7*p^2)) = 1;
        
        
        Data_not_Obs = Data_Obs == 0;
        ind = find(Data_not_Obs == 1);
        t = randperm(length(ind));
        Data_test = Data_not_Obs;
        Data_test(ind(t(1:0.5*length(t)))) = 0;
        
        snr = SNR(noise_ind);
        
        
        % find the noise level to get desired SNR
        
        temp_vec = linspace(0.001,1,1000);
        for i = 1:length(temp_vec)
            snrv = 0;
            for noise_sel = 1:100
                Noise = temp_vec(i)/sqrt(p)*randn(p);
                snrv = snrv +  norm(Lstar,'fro')/norm(Noise,'fro');
            end
            if snrv/100 <= snr
                break;
            end
        end
        noiseSize= temp_vec(i);
        
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
        
        
        
        % compute empirical expectation over 10 trials
        for iter = 1:num_iter
            
            
            Data_Obs = zeros(p);
            t = randperm(p^2);
            Data_Obs(t(1:0.7*p^2)) = 1;
            
            
            Data_not_Obs = Data_Obs == 0;
            ind = find(Data_not_Obs == 1);
            t = randperm(length(ind));
            Data_test = Data_not_Obs;
            Data_test(ind(t(1:0.5*length(t)))) = 0;
            
            Data = (Lstar + noiseSize/sqrt(p)*randn(p));
            
            
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



