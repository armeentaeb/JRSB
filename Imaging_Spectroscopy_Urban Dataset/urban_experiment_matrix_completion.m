%% Written by Armeen Taeb, California Institute of Technology, December 2016
%% This code is for the hyperspectral imaging matrix completion problem
%% In particular, `false_discovery_NS' and `power_NS' find the false discovery
%% and power of no subsampling approach, and `false_discovery_WS' and `power_WS'
%% find the false discovery and power with subsampling approach
clc; close all; clear all
mydir  = pwd;
idcs   = strfind(mydir,'/');
newdir = mydir(1:idcs(end)-1); 
addpath(strcat(newdir,'/Solvers'));


% process data
load('Urban_F210');
data_mat = reshape(Y,210,307,307);
Y = reshape(data_mat(:,90:119,240:264),210,30*25);
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


% model parameters
numTrials = 100;
num_bags = 50;
snr_vec = [10^20];
p = size(Data,2);
samp_rate = 0.1;
alpha = 0.7;
% setup vectors




lam_vector1 = [linspace(0.01,0.1,10) linspace(0.2,30,90)];
lam_vector2 = [linspace(0.01,0.1,10) linspace(0.2,40,90)];
lam_vector = [linspace(0.1,100,100)];
lam_vector = 0.79;%[linspace(0.01,1,20) linspace(1.1,50,50)];


validation_error = zeros(length(lam_vector),numTrials);
power_NS = zeros(length(lam_vector),numTrials);
false_discovery_NS = zeros(length(lam_vector),numTrials);

rank_regular = zeros(length(lam_vector),numTrials);
rank_stability = zeros(length(lam_vector),numTrials);
power_stability = zeros(length(lam_vector),numTrials);
true_signal_stability = zeros(size(Data,2),length(lam_vector));
noise_signal_stability = zeros(size(Data,2),length(lam_vector));
singular_regular = zeros(size(Data,2),length(lam_vector),numTrials);
rank_stability_7 = zeros(length(lam_vector),numTrials);
error = zeros(length(lam_vector),numTrials);
false_discovery_WS = zeros(length(lam_vector),numTrials);
power_WS = zeros(length(lam_vector),numTrials);






 
for trial_ind = 1:numTrials
    
    
    snr = snr_vec(1);
    Data_train = Data;
    t0 = 1-samp_rate;
    A=(rand(size(Data,1),size(Data,2))>t0);
    train_data_locs = A;
    ind = find(A == 0);
    temp = randperm(length(ind));
    test_data_locs = (A ==0);
    test_data_locs(ind(temp(floor(length(ind)*0.05):end))) = 0;

    Z = norm(Data,'fro')/snr*1/2*1/max(sqrt(size(Data)))*randn(size(Data));
    Y = Data + Z;
   
    
    
    % sweep over lambda
    lam_vector = 29;%[linspace(0.01,1,20) linspace(1.1,50,50)];
    for lam_ind = 1:length(lam_vector)
        lambda = lam_vector(lam_ind);
        % run ALS without subsampling
        [rank_regular(lam_ind,trial_ind),proj_matrix,~,A,low_rank_estimate] = matrix_completion_ALS(Y,train_data_locs,20,lambda);
        lambda = lam_vector(lam_ind);
        if rank_regular(lam_ind,trial_ind) == 0
            break;
        end
        
        % find false discoveries and power for no subsampling
        singular_regular(1:p,lam_ind,trial_ind) = svd(low_rank_estimate);
        false_discovery_NS(lam_ind,trial_ind) = trace((perp_dirs)*proj_matrix);
        power_NS(lam_ind,trial_ind) = trace((eye(p)-perp_dirs)*proj_matrix)/trace((eye(p)-perp_dirs));
        validation_error(lam_ind,trial_ind) = norm((low_rank_estimate-Data').*test_data_locs','fro')/(sum(sum(test_data_locs)));
        
        fprintf('--------------------------------------------------------------\n')
        fprintf('this is SNR ind %d\n', trial_ind)
        fprintf('this is lambda %d\n',lam_vector(lam_ind))
        fprintf('the rank of regular is %d\n',rank_regular(lam_ind,trial_ind))
        fprintf('the validation error is %f\n', validation_error(lam_ind,trial_ind))
        fprintf('false discovery is is %f\n',false_discovery_NS(lam_ind,trial_ind))
        fprintf('power is is %f\n',power_NS(lam_ind,trial_ind))
        
        
        avg_proj_matrix_U = zeros(size(Data,2));
        avg_proj_matrix_V = zeros(size(Data,1));
        
%         
%         % sweep over the number of bags
%         for bag_counter = 1:num_bags
%             
%             % obtain one partition of data
%             data_subsampled_1 = train_data_locs;
%             ind = find(train_data_locs == 1);
%             t = randperm(length(ind));
%             data_subsampled_1(ind(t(floor(length(t)/2)+1:end))) = 0;
%             [~,output_proj_stab_U,output_proj_stab_V,~,~] = matrix_completion_ALS(Y,data_subsampled_1,20,lambda);
%             avg_proj_matrix_U = avg_proj_matrix_U  + output_proj_stab_U;
%             avg_proj_matrix_V = avg_proj_matrix_V  + output_proj_stab_V;
%             
%             % obtain the other partion of data
%             data_subsampled_2 = train_data_locs;
%             data_subsampled_2(ind(t(1:floor(length(t)/2)))) = 0;
%             [~,output_proj_stab_U,output_proj_stab_V,~,~] = matrix_completion_ALS(Y,data_subsampled_2,20,lambda);
%             
%             avg_proj_matrix_U = avg_proj_matrix_U  + output_proj_stab_U;
%             avg_proj_matrix_V = avg_proj_matrix_V  + output_proj_stab_V;
%         end
%         
%         [col_space,D_U,~] = svd(avg_proj_matrix_U/(num_bags*2));
%         [row_space,D_V,~] = svd(avg_proj_matrix_V/(num_bags*2));
%         
%         
%         % compute stability selection false discovery and power
%         rank_stability_7(lam_ind,trial_ind) =   min(length(find(diag(D_U)>alpha)),length(find(diag(D_V)>alpha)));
%         proj_matrix = col_space(:,1:rank_stability_7(lam_ind,trial_ind))*col_space(:,1:rank_stability_7(lam_ind,trial_ind))'; 
%         false_discovery_WS(lam_ind,trial_ind) =trace((perp_dirs)*proj_matrix);
%         power_WS(lam_ind,trial_ind) = trace((eye(p)-perp_dirs)*proj_matrix)/trace((eye(p)-perp_dirs));
%         fprintf('the rank of stability alphha 0.7 is %d\n',rank_stability_7(lam_ind,trial_ind))
%         fprintf('the number of false discoveries with alpha 0.7 is %f\n', false_discovery_WS(lam_ind))
%         fprintf('the power with alpha 0.7 is %f\n', power_WS(lam_ind))
%         
       
    end
end
