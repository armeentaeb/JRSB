%% Written by Armeen Taeb, California Institute of Technology, Oct 2018
%% This is a code generates quantities for Figure 5b in the linear measurement setting
%% In particular, `vanilla_FD' number of false discoveries
%% for top 10 ranks with the no-subsampling procedure and `WS_top_projection'
%% finds the number of false discoveries for top 10 ranks with WS-subsampling
%% procedure
close all;clc;clear all;
root = pwd;
addpath(strcat(root,'/Solvers'));
addpath(strcat(root,'/Solvers/cvx'));

p = 70;
num_bags = 50;
SNR = linspace(1,5,5);
rank_vector = [1,2,3,4];
lam_vector = linspace(p^2*0.005,p^2*0.2,100);
num_iter = 100;
rank_const = 10;
alpha_thresh = 0.7;

total_configs = length(rank_vector)*length(SNR);
NS_FD = zeros(num_iter,total_configs);
WS_FD = zeros(num_iter,total_configs);
NS_power = zeros(num_iter,total_configs);
WS_power = zeros(num_iter,total_configs);
incoherence_val = zeros(num_iter,total_configs);

t_train = 0.3;
t_test = 0.15;
config = 0;

load('results_linear_Gaussian_Measurement_large_scale')
config = 10;
for rank_ind = 3:length(rank_vector)
    
    rank = rank_vector(rank_ind);
    rankl = floor(rank*1/2);
    
    % generate the low rank matrix
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
        config = config+1;
        
        snr = SNR(noise_ind);
        
        
        A = randn(p^2*(t_train+t_test),p^2);
        temp_vec = linspace(0.1,10,100);
        for i = 1:length(temp_vec)
            snrv = 0;
            for noise_sel = 1:100
                y = zeros(p^2,1);
                x =  A*reshape(Lstar,p^2,1);
                z = temp_vec(i)*randn(size(A,1),1);
                snrv = snrv+ norm(x)/norm(z);
            end
            
            if snrv/100 < snr
                break;
            end
        end
        
        noiseSize = temp_vec(i);
        y = x + z;
        
        % fix training and test dataset
        temp = randperm(size(A,1));
        A_train = A(temp(1:floor(size(A,1)*t_train)),:);
        y_train = y(temp(1:floor(size(A,1)*t_train)),1);
        A_test = A(temp(floor(size(A,1)*t_train)+1:end),:);
        y_test = y(temp(floor(size(A,1)*t_train)+1:end),1);
        
        validation = 100000;
        for i = 1:length(lam_vector)
            
            lambda = lam_vector(i);
            
            [low_rank_estimate,rk] = gaussian_measurement_norm_large_scale(y_train,A_train,rank_const,p,p,lambda);
            test_error(i) = norm(A_test*reshape(low_rank_estimate,p^2,1)-y_test,'fro');
            if test_error(i) < validation
                validation = test_error(i);
            else
                break;
            end
        end
        [~,ind] = min(test_error);
        lambda = lam_vector(ind) ;
        
        
        
        for iter = 1:100
            
            A_train = randn(p^2*t_train,p^2);
            x =  A_train*reshape(Lstar,p^2,1);
            z = noiseSize*randn(size(A_train,1),1);
            y_train = x+z;
            
            
            
            % find desired noise level
            
            [low_rank_estimate_van,rank_reg(config,iter)] = gaussian_measurement_norm_large_scale(y_train,A_train,rank_const,p,p,lambda);
            
            [U D V] = svd(low_rank_estimate_van);
            
            temp = ((p-rank)^2 - ...
                trace(U(:,rank_reg(config,iter)+1:end)*U(:,rank_reg(config,iter)+1:end)'*perp_dirs_U)* ...
                trace(V(:,rank_reg(config,iter)+1:end)*V(:,rank_reg(config,iter)+1:end)'*perp_dirs_V));
            
            NS_FD(iter,config) = NS_FD(iter,config) + temp;
            NS_power(iter,config) = NS_power(iter,config) + (2*p*rank_reg(config,iter)-rank_reg(config,iter)^2-temp)/(2*p*rank-rank^2);
            
            
            % compute average projection matrix
            avg_proj_matrix_U = zeros(p);
            avg_proj_matrix_V = zeros(p);
            avg_tangent_space = zeros(p^2);
            row_bags = cell(num_bags*2,1);
            col_bags = cell(num_bags*2,1);
            
            for j = 1:num_bags
                t = randperm(size(A_train,1));
                y_sub_1 = y_train(t(1:floor(length(t)/2)),1);
                A_sub_1 = A_train(t(1:floor(length(t)/2)),:);
                [low_rank_estimate_van,rk] = gaussian_measurement_norm_large_scale(y_sub_1,A_sub_1,rank_const,p,p,lambda);
                [U D V] = svd(low_rank_estimate_van); output_proj_stab_U = U(:,1:rk)*U(:,1:rk)';
                output_proj_stab_V = V(:,1:rk)*V(:,1:rk)';
                
                
                %[~,output_proj_stab_U,output_proj_stab_V,~,~] = gaussian_measurement_norm(y_sub_1,A_sub_1,lambda);
                %max(max(abs(avg_proj_matrix_U/max((2*j-2),1) -(avg_proj_matrix_U  + output_proj_stab_U)/max((2*j-1),1))))
                
                avg_proj_matrix_U = avg_proj_matrix_U  + output_proj_stab_U;
                avg_proj_matrix_V  = avg_proj_matrix_V  + output_proj_stab_V;
                row_bags{2*j-1} = output_proj_stab_V;
                col_bags{2*j-1} = output_proj_stab_U;
                
                y_sub_2 = y_train(t(floor(length(t)/2)+1:end),1);
                A_sub_2 = A_train(t(floor(length(t)/2)+1:end),:);
                %[~,output_proj_stab_U,output_proj_stab_V,~,~] = gaussian_measurement_norm(y_sub_2,A_sub_2,lambda);
                [low_rank_estimate_van,rk] = gaussian_measurement_norm_large_scale(y_sub_2,A_sub_2,rank_const,p,p,lambda);
                [U D V] = svd(low_rank_estimate_van); output_proj_stab_U = U(:,1:rk)*U(:,1:rk)';
                output_proj_stab_V = V(:,1:rk)*V(:,1:rk)';
                
                
                
                avg_proj_matrix_U = avg_proj_matrix_U  + output_proj_stab_U;
                avg_proj_matrix_V = avg_proj_matrix_V  + output_proj_stab_V;
                row_bags{2*j} = output_proj_stab_V;
                col_bags{2*j} = output_proj_stab_U;
            end
            
            [U, D_U ,~] = svd(avg_proj_matrix_U/(num_bags*2));
            [V, D_V ,~] = svd(avg_proj_matrix_V/(num_bags*2));
            
            
            
            k = min(length(find(diag(D_U)>alpha_thresh)),length(find(diag(D_V)>alpha_thresh)));
            rank_stab(config,iter) = k;
            
            % compute false discoveries
            temp = ((p-rank)^2 - ...
                trace(U(:,rank_stab(config,iter)+1:end)*U(:,rank_stab(config,iter)+1:end)'*perp_dirs_U)* ...
                trace(V(:,rank_stab(config,iter)+1:end)*V(:,rank_stab(config,iter)+1:end)'*perp_dirs_V));
            
            WS_FD(iter,config) = WS_FD(iter,config) + temp;
            WS_power(iter,config) =  WS_power(iter,config)+(2*p*rank_stab(config,iter)-rank_stab(config,iter)^2-temp)/(2*p*rank-rank^2);
            
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
