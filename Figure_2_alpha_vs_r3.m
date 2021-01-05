%% Written by Armeen Taeb, California Institute of Technology, December 2016
%% This is a code that generates the quantities for Figure 2
%% In particular, the matrix ``alpha" stores the values of 
%% $\sigma_min(P_TP_avgP_T)$ for $r3 = 1,2,...p$

root = pwd;
addpath(strcat(root,'/Solvers'));
addpath(strcat(root,'/Solvers/cvx'));

% setup model parameters
p = 70;
rank = 10;
rankl = 3;
SNR = [50 1.2 0.8 0.4];
num_bags = 50;

% setup arrays

alpha = zeros(length(lam_vector1),rank,length(SNR));
lambda_CV = zeros(length(lam_vector1),1);
false_discovery_stabiliy = zeros(length(lam_vector1),rank,length(SNR));
validation_performance = zeros(length(lam_vector1),1);

% the range of lambda values for each SNR
lam_vector1 = [linspace(0.001,0.1,40) linspace(0.1,5,40)];
lam_vector2 = [linspace(0.001,0.1,40) linspace(0.1,0.5,40)];
lam_vector3 = [linspace(0.001,0.1,40) linspace(0.1,0.5,40)];
lam_vector4 = [linspace(0.001,0.1,40) linspace(0.1,0.5,40)];



[U,~,~] = svd(randn(p));
[V,~,~] = svd(randn(p));


% generate the population matrix
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

[Ut,~,Vt] = svd(Lstar);
perp_dirs_U = Ut(:,rank+1:end)*Ut(:,rank+1:end)';
perp_dirs_V = Vt(:,rank+1:end)*Vt(:,rank+1:end)';

for noise_ind = 1:length(SNR)
    if noise_ind == 1
        lam_vector = lam_vector1;
    elseif noise_ind == 2
        lam_vector = lam_vector2;
    elseif noise_ind == 3
        lam_vector = lam_vector3;
    elseif noise_ind == 4
        lam_vector = lam_vector4;
    end
    fprintf('this is noise index %d\n',noise_ind)
    
    
    % find the noise level to achieve desired SNR
    temp_vec = linspace(0.001,1,1000);
    for i = 1:length(temp_vec)
        Noise = temp_vec(i)/sqrt(p)*randn(p);
        if norm(Lstar,'fro')/norm(Noise,'fro') < SNR(noise_ind)
            break;
        end
    end
    Data = (Lstar + Noise);
    
    
    % create training and testing dataset
    t0 = 0.65;
    Data_Obs = (rand(p) > t0);
    Data_not_Obs = Data_Obs == 0;
    ind = find(Data_not_Obs == 1);
    t = randperm(length(ind));
    Data_test = Data_not_Obs;
    Data_test(ind(t(floor(p^2*0.3)+1:end))) = 0;
    
    
    % sweep over lambda
    for i = 1:length(lam_vector)
        
        % run matrix completion on full data
        lambda = lam_vector(i);
        [~,~,~,A,low_rank_estimate] = matrix_completion_nuclear_norm(Data',Data_Obs',size(Lstar,1),lambda);
        validation_performance(i) = norm((low_rank_estimate-Data).*Data_test,'fro');
        
        avg_proj_matrix_U  = zeros(p);
        avg_proj_matrix_V  = zeros(p);
        avg_tangent_space = zeros(p^2);
        row_bags = cell(num_bags*2,1);
        col_bags = cell(num_bags*2,1);
        
        % compute subsampled tangent spaces
        for j = 1:num_bags
            
            % generate a partition of data
            data_subsampled_1 = Data_Obs;
            ind = find(Data_Obs == 1);
            t = randperm(length(ind));
            data_subsampled_1(ind(t(floor(length(t)/2)+1:end))) = 0;
            
            % execute nuclear norm minimization on one partition of data
            [~,output_proj_stab_U,output_proj_stab_V,~,~] = matrix_completion_nuclear_norm(Data',data_subsampled_1',size(Lstar,1),lambda);
            avg_proj_matrix_U = avg_proj_matrix_U  + output_proj_stab_U;
            avg_proj_matrix_V  = avg_proj_matrix_V  + output_proj_stab_V;
            row_bags{2*j-1} = output_proj_stab_V;
            col_bags{2*j-1} = output_proj_stab_U;
            

            data_subsampled_2 = Data_Obs;
            data_subsampled_2(ind(t(1:floor(length(t)/2)))) = 0;
           
            % execute nuclear norm minimization on another partition of data
            [~,output_proj_stab_U,output_proj_stab_V,~,~] = matrix_completion_nuclear_norm(Data',data_subsampled_2',size(Lstar,1),lambda);
            avg_proj_matrix_U = avg_proj_matrix_U  + output_proj_stab_U;
            avg_proj_matrix_V = avg_proj_matrix_V  + output_proj_stab_V;
            row_bags{2*j} = output_proj_stab_V;
            col_bags{2*j} = output_proj_stab_U;
        end
        
        [U, D_U ,~] = svd(avg_proj_matrix_U/(num_bags*2));
        [V, D_V ,~] = svd(avg_proj_matrix_V/(num_bags*2));
        
        % compute average projection matrix
        P_avg = zeros(p^2);
        for j = 1:num_bags
            P_avg  =  P_avg  + kron((eye(p)-col_bags{2*j-1}),(eye(p)-row_bags{2*j-1}));
            P_avg  =  P_avg  + kron((eye(p)-col_bags{2*j}),(eye(p)-row_bags{2*j}));
        end
        
        % find the value of $\sigma_min(P_TP_avgP_T)$ for different r3
        % values
        for k = 1:p
            P_T = eye(p^2) - kron(U(:,k+1:end)*U(:,k+1:end)',V(:,k+1:end)*V(:,k+1:end)');
            alpha(i,k,noise_ind) = 1-norm(P_T*P_avg/(2*num_bags)*P_T);
        end
        
    end
    
    [~,ind] = min(validation_performance);
    lambda_CV(noise_ind) = lam_vector(ind);
end








