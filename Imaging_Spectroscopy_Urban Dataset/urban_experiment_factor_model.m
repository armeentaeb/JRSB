%% Written by Armeen Taeb, California Institute of Technology, December 2016
%% This code is for the hyperspectral imaging factor model analysis
%% In particular, `false_discovery_NS' and `power_NS' find the false discovery
%% and power of no subsampling approach, and `false_discovery_WS' and `power_WS'
%% find the false discovery and power with subsampling approach

mydir  = pwd;
idcs   = strfind(mydir,'/');
newdir = mydir(1:idcs(end)-1);

addpath(strcat(newdir,'/Solvers'));

load('Urban_F210');
data_mat = reshape(Y,210,307,307);
Y = reshape(data_mat(:,90:119,240:264),210,30*25);

%% read data and the ground-truth
load('end5_groundTruth.mat')
num_endmem = 3;
[U,~,~] = svd(M(:,[1,4,2]));
perp_dirs = U(:,num_endmem+1:end)*U(:,num_endmem+1:end)';
Y = Y(SlectBands,:)/max(max(Y));
Data = Y';



sample_cov = cov(Data);
var_scaling = diag(diag(sample_cov).^(-1/2));
Data = Data*var_scaling;
Data = Data(:,:);
num_bags = 50;


%% setup parameters
lam_vector = [linspace(1,8,50)];
noise = [5];
p = size(Data,2);
alpha_vec = 0.7;

%% setup vectors
validation_error = zeros(length(alpha_vec),1);
rank_regular = zeros(length(alpha_vec),1);
rank_stability = zeros(length(alpha_vec),1);
false_discovery_stability = zeros(length(alpha_vec),1);
power_stability = zeros(length(alpha_vec),1);
rank_stability_7 = zeros(length(alpha_vec),1);
false_discovery_WS = zeros(length(alpha_vec),length(noise));
stability_error_7 = zeros(length(alpha_vec),length(noise));
power_stability_WS = zeros(length(alpha_vec),length(noise));
false_discovery = 0;
power = 0;
false_discovery_bound = zeros(length(alpha_vec),length(noise));
[Ustar,~,~] = svd(perp_dirs);
validation = length(lam_vector);

% add noise to data to achieve SNR ~= 0.78
tr_tt_perm = randperm(size(Data,1));
Z = norm(Data,'fro')/noise*1/2*1/max(sqrt(size(Data)))*randn(size(Data));
Y = Data + Z;
Y_train = Y(tr_tt_perm(1:floor(length(tr_tt_perm)*8/10)),:);
Y_test = Y(tr_tt_perm(floor(length(tr_tt_perm)*8/10)+1:length(tr_tt_perm)),:);
q = 0;
tau_star = 0;


% sweeping over lambda to choose cross-validated choice
for lam_ind = 1:length(lam_vector)
    lambda = lam_vector(lam_ind);
    [F,rank_regular(lam_ind),~] = factor_model(Y_train,Y_test,lambda);
    Sigma_test = diag(diag(cov(Y_train)).^(-1/2))*cov(Y_test)* diag(diag(cov(Y_train)).^(-1/2));
    validation(lam_ind)=norm(Sigma_test-F{1}^(-1),'fro');
    D = F{1}+F{2};
    [U,dp,~] = svd(F{1}^(-1)-D^(-1));
    rank_regular(lam_ind) = length(find(diag(dp)>10^(-3)));
    proj_matrix = U(:,1:rank_regular(lam_ind))*U(:,1:rank_regular(lam_ind))';
    if rank_regular(lam_ind) == 0
        break;
    end
    
end
[~,ind] = min(validation);
lambda = lam_vector(ind);

false_discovery_NS = 0;
power_NS = 0;

% compute empirical estimate over 50 iterations
for iter = 1:100
    
    sprintf('this is iter %d \n', iter)
    
    avg_proj_matrix_U = zeros(size(Data,2));
    avg_proj_matrix_V = zeros(size(Data,1));
    Z = norm(Data,'fro')/noise*1/2*1/max(sqrt(size(Data)))*randn(size(Data));
    Y = Data + Z;
    Y_train = Y(tr_tt_perm(1:floor(length(tr_tt_perm)*8/10)),:);
    Y_test = Y(tr_tt_perm(floor(length(tr_tt_perm)*8/10)+1:length(tr_tt_perm)),:);
    
    [F,rank_regular(lam_ind),valid] = factor_model(Y_train,Y_test,lambda);
    D = F{1}+F{2};
    [U,dp,~] = svd(F{1}^(-1)-D^(-1));
    rank_regular_CV = rank_regular(ind);
    rank_regular(lam_ind) = length(find(diag(dp)>10^(-3)));
    k = rank_regular(lam_ind);
    temp = ((p-3)^2 - trace(U(:,k+1:end)*U(:,k+1:end)'*perp_dirs)* trace(U(:,k+1:end)*U(:,k+1:end)'*perp_dirs));
    false_discovery_NS = false_discovery_NS + temp;
    power_NS = power_NS + (2*p*k-k^2-temp)/(2*p*k-k^2);
    
    
    % bagging procedure
    for bag_counter = 1:num_bags
        
        % partition data
        t = randperm(size(Y_train,1));
        Y_train_sub = Y_train(t(1:length(t)/2),:);
        % run factor model analysis on one partition of data
        [F,~,~] = factor_model(Y_train_sub,Y_test,lambda);
        D = F{1}+F{2};
        [U,dp,~] = svd(F{1}^(-1)-D^(-1));
        rk = length(find(dp>10^(-3)));
        output_proj_stab_U = U(:,1:rk)*U(:,1:rk)';
        
        
        
        avg_proj_matrix_U = avg_proj_matrix_U  + output_proj_stab_U;
        Y_train_sub = Y_train(t(length(t)/2+1:end),:);
        
        % run factor model analysis on the other partition of data
        [F,~,~] = factor_model(Y_train_sub,Y_test,lambda);
        D = F{1}+F{2};
        [U,dp,~] = svd(F{1}^(-1)-D^(-1));
        rk = length(find(dp>10^(-3)));
        output_proj_stab_U = U(:,1:rk)*U(:,1:rk)';
        avg_proj_matrix_U = avg_proj_matrix_U  + output_proj_stab_U;
        
        
    end
    
    
    [col_space,D_U,~] = svd(avg_proj_matrix_U/(num_bags*2));
    
    % compute the number of false discoveries and power of subspace
    % stability selection
    rank_stability_7 = length(find(diag(D_U) > alpha_vec));
    k = rank_stability_7;
    proj_matrix = col_space(:,1:rank_stability_7)*col_space(:,1:rank_stability_7)';
    temp = ((p-3)^2 - trace(U(:,k+1:end)*U(:,k+1:end)'*perp_dirs)* trace(U(:,k+1:end)*U(:,k+1:end)'*perp_dirs));
    false_discovery_WS =false_discovery_WS + temp;
    power_stability_WS = power_stability_WS + (2*p*k-k^2-temp)/(2*p*k-k^2);
    
end

false_discovery_NS = false_discovery_NS/100;
power_NS = power_NS/100;
false_discovery_WS = false_discovery_WS/100;
power_stability_WS =  power_stability_WS/100;




