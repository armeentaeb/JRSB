%% Written by Armeen Taeb, California Institute of Technology, December 2016
%% This is a code that generates the quantities for Figure 3
%% In particular, `NS_top_projection' finds the number of false discoveries
%% for top 10 ranks with the no-subsampling procedure and `WS_top_projection'
%% finds the number of false discoveries for top 10 ranks with WS-subsampling
%% procedure
clc;close all;clear all;
root = pwd;
addpath(strcat(root,'/Solvers'));

% setup model parameters
p = 70;
num_bags = 50;
SNR = [1.5 2 2.5 3];
inco_vector = 0.6;
obs_prop = [0.65]; obs = 0.65;
config = 0;
rank = 10;
num_iter = 100;
alpha_thresh = 0.7;
lam_vector = linspace(0.1,1,30);
total_configs = length(SNR);

% setting up all the data vectors
NS_FD = zeros(total_configs,num_iter);
WS_FD = zeros(total_configs,num_iter);
validation_performance = zeros(length(lam_vector),total_configs);
lambda_CV = zeros(length(lam_vector),total_configs);



% find row/column space with high incoherence
[U D V] = svd(randn(p));
Lstar = zeros(p);
for i = 1:3
    sigma = 1;
    Lstar = Lstar + sigma*U(:,i)*U(:,i)';
end

for i = 4:8
    sigma =    0.5;
    Lstar = Lstar + sigma*U(:,i)*U(:,i)';
end
for i = 9:10
    sigma =    0.1;
    Lstar = Lstar + sigma*U(:,i)*U(:,i)';
end

[U,~,V] = svd(Lstar);
perp_dirs_U = U(:,rank+1:end)*U(:,rank+1:end)';
perp_dirs_V = V(:,rank+1:end)*V(:,rank+1:end)';



for noise_ind = 1:length(SNR)
    
    % find snr level and observation rate
    snr = SNR(noise_ind);
    config = config+1;
    
    
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
    Data = Lstar + Noise;
    
    
    
    % select observed portion of data
    Data_Obs = (rand(p) > obs);
    Data_not_Obs = Data_Obs == 0;
    ind = find(Data_not_Obs == 1);
    t = randperm(length(ind));
    Data_test = Data_not_Obs;
    Data_test(ind(t(floor(p^2*0.3)+1:end))) = 0;
    
    % compute empirical averages over 10 trials
    for lam_ind = 1:length(lam_vector)
        %lam_ind
        lambda = lam_vector(lam_ind);
        [~,~,~,A,low_rank_estimate] = matrix_completion_nuclear_norm(Data',Data_Obs',size(Lstar,1),lambda);
        [U,~,V] = svd(low_rank_estimate);
        % validation performance
        validation_performance(lam_ind,config) = norm((low_rank_estimate-Data).*Data_test,'fro');
    end
    [~,ind] = min(validation_performance(:,config));
    lambda = lam_vector(ind);
    
    
    % without sub-sampling matrix completion
    
    
    
    %% now compute empirical average
    for iter = 1:num_iter
        
        
        
        
        
%         temp_vec = linspace(0.001,1,1000);
%         for i = 1:length(temp_vec)
%             Noise = temp_vec(i)/sqrt(p)*randn(p);
%             if norm(Lstar,'fro')/norm(Noise,'fro') < snr
%                 break;
%             end
%         end
        Data = (Lstar + noiseSize/sqrt(p)*randn(p));
        
        
        
        
        % select observed portion of data
        Data_Obs = (rand(p) > obs);
        Data_not_Obs = Data_Obs == 0;
        ind = find(Data_not_Obs == 1);
        t = randperm(length(ind));
        Data_test = Data_not_Obs;
        Data_test(ind(t(floor(p^2*0.3)+1:end))) = 0;
        
        
        % find noise level to achieve desired SNR level
        
        
        [~,~,~,A,low_rank_estimate] = matrix_completion_nuclear_norm(Data',Data_Obs',size(Lstar,1),lambda);
        [U,~,V] = svd(low_rank_estimate);
        rk = length(find(svd(low_rank_estimate)>10^(-3)));
        NS_FD(config,iter) =  ((p-rank)^2 - ...
            trace(U(:,rk+1:end)*U(:,rk+1:end)'*perp_dirs_U)* ...
            trace(V(:,rk+1:end)*V(:,rk+1:end)'*perp_dirs_V));
        
        
        
        %                 % perform subsampling
        avg_proj_matrix_U  = zeros(p);
        avg_proj_matrix_V  = zeros(p);
        avg = 0;
        avg_disc = 0;
        P_avg = zeros(p^2);
        for j = 1:num_bags
            
            % partition the data
            data_subsampled_1 = Data_Obs;
            ind = find(Data_Obs == 1);
            t = randperm(length(ind));
            data_subsampled_1(ind(t(floor(length(t)/2)+1:end))) = 0;
            
            % run nuclear norm estimator on one partition of
            % data
            
            [~,output_proj_stab_U,output_proj_stab_V,~,L] = matrix_completion_nuclear_norm(Data',data_subsampled_1',size(Lstar,1),lambda);
            
            P_avg = P_avg+kron(eye(p)-output_proj_stab_U,eye(p)-output_proj_stab_V);
            
            avg = avg + trace((eye(p)-output_proj_stab_U)*perp_dirs_U)* ...
                trace((eye(p)-output_proj_stab_V)*perp_dirs_V);
            
            avg_disc = avg_disc + 2*p*trace(output_proj_stab_U)-trace(output_proj_stab_U)^2-...
                (trace(output_proj_stab_U*(perp_dirs_U))*trace(output_proj_stab_V) + ...
                trace(output_proj_stab_V*(perp_dirs_V))*trace(output_proj_stab_U) - ...
                trace(output_proj_stab_U*(perp_dirs_U))*trace(output_proj_stab_V*(perp_dirs_V)));
            
            trace((output_proj_stab_U))*trace(output_proj_stab_V);
            avg_proj_matrix_U = avg_proj_matrix_U  + output_proj_stab_U;
            avg_proj_matrix_V  = avg_proj_matrix_V  + output_proj_stab_V;
            
            data_subsampled_2 = Data_Obs;
            data_subsampled_2(ind(t(1:floor(length(t)/2)))) = 0;
            
            % run nuclear norm estimator on the other partition of
            % data
            [~,output_proj_stab_U,output_proj_stab_V,~,L] = matrix_completion_nuclear_norm(Data',data_subsampled_2',size(Lstar,1),lambda);
            P_avg = P_avg+kron(eye(p)-output_proj_stab_U,eye(p)-output_proj_stab_V);
            
            avg = avg + trace((eye(p)-output_proj_stab_U)*perp_dirs_U)* ...
                trace((eye(p)-output_proj_stab_V)*perp_dirs_V);
            avg_disc = avg_disc + 2*p*trace(output_proj_stab_U)-trace(output_proj_stab_U)^2-...
                (trace(output_proj_stab_U*(perp_dirs_U))*trace(output_proj_stab_V) + ...
                trace(output_proj_stab_V*(perp_dirs_V))*trace(output_proj_stab_U) - ...
                trace(output_proj_stab_U*(perp_dirs_U))*trace(output_proj_stab_V*(perp_dirs_V)));
            avg_proj_matrix_U = avg_proj_matrix_U  + output_proj_stab_U;
            avg_proj_matrix_V = avg_proj_matrix_V  + output_proj_stab_V;
            
        end
        
        [U, D_U ,~] = svd(avg_proj_matrix_U/(num_bags*2));
        [V, D_V ,~] = svd(avg_proj_matrix_V/(num_bags*2));
        
        
        
        k = 0;
        alpha_val = 1;
        while alpha_val > alpha_thresh
            k = k+1;
            P_T = eye(p^2) - kron(U(:,k+1:end)*U(:,k+1:end)',V(:,k+1:end)*V(:,k+1:end)');
            alpha_val = 1-norm(P_T*P_avg/(2*num_bags)*P_T);
            alpha(k,noise_ind) = alpha_val;
        end
        
        
        
        
        
        WS_FD(config,iter) =  ((p-rank)^2 - ...
            trace(U(:,k+1:end)*U(:,k+1:end)'*perp_dirs_U)* ...
            trace(V(:,k+1:end)*V(:,k+1:end)'*perp_dirs_V));
        
    end
    
    
    save('results_Table1.mat')
end


close all;plot(lam_vector,mean(squeeze(WS_FD(4,:,1,:)),2))
hold on;
plot(mean(squeeze(NS_FD(4,:,1,:)),2))
