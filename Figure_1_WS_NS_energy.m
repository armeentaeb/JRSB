%% Written by Armeen Taeb, California Institute of Technology, December 2016
%% This is a code that generates the quantities for Figure 1
%% In particular, NS_energy_noise is the no-subampling energy in null directions
%% WS_energy_noise is with-subsampling energy in false directions

root = pwd;
addpath(strcat(root,'/Solvers'));
addpath(strcat(root,'/Solvers/cvx'));

% setup model parameters
p = 70;
rank = 10;
rankl = 3;
num_bags = 50;
lam_vector = linspace(0.01,1,30);
snr = [1.6 0.8];

% setup arrays
validation_performance = zeros(length(lam_vector),1);
NS_energy_noise = zeros(length(lam_vector),length(snr),20);
WS_energy_noise = zeros(length(lam_vector),length(snr),20);
rank_regular = zeros(length(lam_vector),length(snr));
lambda_CV = zeros(length(lam_vector),length(snr));


% create the population low rank matrix
[U,~,~] = svd(randn(p));
[V,~,~] = svd(randn(p));

Lstar = zeros(p);
for i = 1:rankl
    sigma = 1;
    Lstar = Lstar + sigma*U(:,i)*V(:,i)';
end

for i = rankl+1:8
    sigma = 0.5;
    Lstar = Lstar + sigma*U(:,i)*V(:,i)';
end

for i = 9:10
    sigma = 0.1;
    Lstar = Lstar + sigma*U(:,i)*V(:,i)';
end



% Create train and test data
t0 = 0.65;
Data_Obs = (rand(p) > t0);
Data_not_Obs = Data_Obs == 0;
ind = find(Data_not_Obs == 1);
t = randperm(length(ind));
Data_test = Data_not_Obs;
Data_test(ind(t(floor(p^2*0.3)+1:end))) = 0;

load('Figure_1_results.mat')
for ind = 1:length(snr)
    
    ind = 2;
    fprintf('this is snr %d\n',snr(ind))
    snr_val = snr(ind);
    
    for iter = 10:20
        
        
        % find the noise level to have desired SNR
        temp_vec = linspace(0.001,1,1000);
        for i = 1:length(temp_vec)
            Noise = temp_vec(i)/sqrt(p)*randn(p);
            if norm(Lstar,'fro')/norm(Noise,'fro') < snr_val
                break;
            end
        end
        Data = (Lstar + Noise);
        
        
        
        [U,~,V] = svd(Lstar);
        perp_dirs_U = U(:,rank+1:end)*U(:,rank+1:end)';
        perp_dirs_V = V(:,rank+1:end)*V(:,rank+1:end)';
        
        
        % sweep over lambda 
        for i = 1:length(lam_vector)
            
            lambda = lam_vector(i);
            [rank_regular(i,ind),~,~,A,low_rank_estimate] = matrix_completion_nuclear_norm(Data',Data_Obs',size(Lstar,1),lambda);
            [U,~,V] = svd(low_rank_estimate);
            validation_performance(i) = norm((low_rank_estimate-Data).*Data_test,'fro');
            rank_est = rank_regular(i,ind);
            
            % energy of N-S
            NS_energy_noise(i,ind,iter) =  ((p-rank)^2 - ...
                trace(U(:,rank_est+1:end)*U(:,rank_est+1:end)'*perp_dirs_U)* ...
                trace(V(:,rank_est+1:end)*V(:,rank_est+1:end)'*perp_dirs_V));
            
            
            
            % perform complementary subsampling
            avg_proj_matrix_U  = zeros(p);
            avg_proj_matrix_V  = zeros(p);
            P_avg = zeros(p^2);
            avg = 0;
            avg_disc = 0;
            
            % iterate over the number of complementary bags
            for j = 1:num_bags
                
                data_subsampled_1 = Data_Obs;
                temp = find(Data_Obs == 1);
                t = randperm(length(temp));
                data_subsampled_1(temp(t(floor(length(t)/2)+1:end))) = 0;
                % solve nuclear norm minimization with one partition of
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
                
                % compute summands of the average projection matrix
                P_avg = P_avg +  kron(output_proj_stab_U,output_proj_stab_U);
                
                
                data_subsampled_2 = Data_Obs;
                data_subsampled_2(temp(t(1:floor(length(t)/2)))) = 0;
                % solve nuclear norm minimization with the other partition
                % of data

                [~,output_proj_stab_U,output_proj_stab_V,~,L] = matrix_completion_nuclear_norm(Data',data_subsampled_2',size(Lstar,1),lambda);
                avg = avg + trace((eye(p)-output_proj_stab_U)*perp_dirs_U)* ...
                    trace((eye(p)-output_proj_stab_V)*perp_dirs_V);
                avg_disc = avg_disc + 2*p*trace(output_proj_stab_U)-trace(output_proj_stab_U)^2-...
                    (trace(output_proj_stab_U*(perp_dirs_U))*trace(output_proj_stab_V) + ...
                    trace(output_proj_stab_V*(perp_dirs_V))*trace(output_proj_stab_U) - ...
                    trace(output_proj_stab_U*(perp_dirs_U))*trace(output_proj_stab_V*(perp_dirs_V)));
                avg_proj_matrix_U = avg_proj_matrix_U  + output_proj_stab_U;
                avg_proj_matrix_V = avg_proj_matrix_V  + output_proj_stab_V;
                % compute summands of the average projection matrix
                P_avg = P_avg +  kron(output_proj_stab_U,output_proj_stab_U);
            end
            
            % compute energy with subsampling
            WS_energy_noise(i,ind,iter) = (p-10)^2-avg/(2*num_bags);
            
        end
        [~,temp] = min(validation_performance);
    end
    lambda_CV(ind) = lam_vector(temp);

end


