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
SNR = [1.6];
inco_vector = 0.6;
obs_prop = [0.65]; obs = 0.65;
config = 0;
rank = 10;
num_iter = 100;

lam_vector = linspace(0.2,0.5,20);
total_configs = length(SNR)*length(inco_vector)*length(obs_prop);

% setting up all the data vectors
NS_FD = zeros(rank,length(lam_vector),total_configs,num_iter);
WS_FD = zeros(rank,length(lam_vector),total_configs,num_iter);
validation_performance = zeros(length(lam_vector),total_configs);
lambda_CV = zeros(length(lam_vector),total_configs);



% find row/column space with high incoherence
U = eye(p);
V = eye(p);
[U,~,~] = svd(U(:,1:rank)*U(:,1:rank)' + randn(p)/25);
[V,~,~] = svd(V(:,1:rank)*V(:,1:rank)' + randn(p)/25);    % setup population model
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
        
        
    
    
    
    
    % compute empirical averages over 10 trials
    
    
    
    %% now compute empirical average
    for iter = 1:num_iter
        iter  
        Data_Obs = (rand(p) > obs);
        Data_not_Obs = Data_Obs == 0;
        ind = find(Data_not_Obs == 1);
        t = randperm(length(ind));
        Data_test = Data_not_Obs;
        Data_test(ind(t(floor(p^2*0.3)+1:end))) = 0;
        Data = (Lstar + noiseSize/sqrt(p)*randn(p));

        
        for lam_ind = 1:length(lam_vector)
            lambda = lam_vector(lam_ind);
            % without sub-sampling matrix completion
            
            [~,~,~,A,low_rank_estimate] = matrix_completion_nuclear_norm(Data',Data_Obs',size(Lstar,1),lambda);
            [U,~,V] = svd(low_rank_estimate);
            rank_regular(noise_ind,lam_ind) = length(find(svd(low_rank_estimate)>10^(-3)));
            validation_performance(lam_ind) = norm((low_rank_estimate-Data).*Data_test,'fro');
            
            % compute the false discoveries in top components of
            % the low rank matrix from N-S
            for rk = 1:rank      
                NS_FD(rk,lam_ind,config,iter) =  ((p-rank)^2 - ...
                    trace(U(:,rk+1:end)*U(:,rk+1:end)'*perp_dirs_U)* ...
                    trace(V(:,rk+1:end)*V(:,rk+1:end)'*perp_dirs_V));
            end
            
            
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
            
            
            
            % compute WS_FD for different ranks
            if norm(avg_proj_matrix_U) > 10^(-6) &&  norm(avg_proj_matrix_V) > 10^(-6)
                for j = 1:rank
                    WS_FD(j,lam_ind,config,iter) =  ((p-rank)^2 - ...
                        trace(U(:,j+1:end)*U(:,j+1:end)'*perp_dirs_U)* ...
                        trace(V(:,j+1:end)*V(:,j+1:end)'*perp_dirs_V));
                end
            else
                for j = 1:rank
                    WS_FD(j,lam_ind,config,iter) =  1000;
                end
            end
        end
        
        save('results_Figure3_newest_1.6.mat')
    end
    for lam_ind = 1:length(lam_vector)

        for rank_ind = 1:rank
            WS_FD_mean(rank_ind,lam_ind,config) = mean(squeeze(WS_FD(rank_ind,lam_ind,config,:)));
            NS_FD_mean(rank_ind,lam_ind,config) = mean(squeeze(NS_FD(rank_ind,lam_ind,config,:)));
            WS_FD_variance(rank_ind,lam_ind,config) = std(squeeze(WS_FD(rank_ind,lam_ind,config,:)));
            NS_FD_variance(rank_ind,lam_ind,config) = std(squeeze(NS_FD(rank_ind,lam_ind,config,:)));
        end
    end
    [~,ind] = min(validation_performance(:,config));
    lambda_CV(config) = ind;
end

 close all
plot(lam_vector,NS_FD_mean(3,:,1),'LineWidth',3)
hold on; plot(lam_vector,WS_FD_mean(3,:,1),'LineWidth',3)
plot(lam_vector(10)*ones(100,1),linspace(40,110,100),'k--','LineWidth',3)
xlabel('\lambda')
ylabel('false discovery')
set(gca,'FontSize',14)
set(gcf,'color','w');
legend('N-S','W-S','CV')



set(gca,'FontSize',16)

plot(lam_vector,WS_FD_mean(3,:,2),'LineWidth',3)
hold on;plot(lam_vector,NS_FD_mean(3,:,2),'LineWidth',3)
plot(lam_vector(10)*ones(100,1),linspace(40,110,100),'k--','LineWidth',3)
set(gca,'FontSize',14)