%% Written by Armeen Taeb, California Institute of Technology, December 2016
%% This code is for showing the boost in Gaussian linear measurements
clear all; clc
addpath('/Users/armeentaeb/Dropbox/JRSB_Data_CODE/Solvers')



%% Problem parameters
p = [200];rank = 6;rank_const =6;rank_star = rank;
snr = [0.15];
n = 1*p;
num_iter = 100;num_bags = 50;
n = 2*p;
alpha_vec = linspace(0.7,0.99,20)';
lam_vector = n*0.01*linspace(1,200,30);
Gauss_str = [30,10];
ranks = 6;
%% generate undelying low-rank matrix

[U,~,V] = svd(randn(p));
Lstar = U(:,1:rank)*diag([120*ones(1,1),100,80,30*ones(1,1),20,10])*V(:,1:rank)';
[Ustar,Dstar,Vstar] = svd(Lstar);
perp_dirs_U = Ustar(:,rank+1:end)*Ustar(:,rank+1:end)';
perp_dirs_V = Vstar(:,rank+1:end)*Vstar(:,rank+1:end)';

%% outer loop parameters
total_bound_2 = zeros(length(alpha_vec),length(Gauss_str));
total_bound_1 = zeros(length(alpha_vec),length(Gauss_str));
fd_stability = zeros(length(alpha_vec),length(Gauss_str));
avg_discovery = zeros(length(alpha_vec),length(Gauss_str));
fd_nonsubsampling = zeros(length(alpha_vec),1);

half_data_basis = zeros((p-rank)^2,length(Gauss_str));
half_data_nobasis = zeros(p^2,length(Gauss_str));

config = 0;
for noise_ind = 1:length(Gauss_str)
    
    config = config+1;
    
    %% Inner Loop parameters
    temp_full =  0;
    temp_data_interactions = zeros(1,length(alpha_vec));
    temp_half_data_basis_dependent =  zeros((p-rank)^2,1);
    temp_stability = zeros(length(alpha_vec),1);
    commutator_indiv = zeros((p-rank)^2,1);
    commutator_tot = zeros(num_iter,1);
    temp_data_interactions_2 = zeros(length(alpha_vec),1)';
    
    
    
    %% compute empirical average on num_iter iterations
    temp_vec = linspace(0.1,100,1000);
    temp_noise = zeros(length(temp_vec),1);
    for iter = 1:num_iter
        for i = 1:length(temp_vec)
            Z = temp_vec(i)*(Gauss_str(noise_ind)*Ustar*(diag(randn(p,1)))*Vstar(:,1:p)'+randn(p));
            Y =  Lstar+Z;
            temp_noise(i) = temp_noise(i)+norm(Lstar)/norm(Z);
        end
    end
    [ind] = find(temp_noise/num_iter <= snr); noise_degree = temp_vec(ind(1));
    
    %% fix training and test dataset
    Data = cell(n,1);
    for j = 1:n
        Z = noise_degree*(Gauss_str(noise_ind)*Ustar*(diag(randn(p,1)))*Vstar(:,1:p)'+randn(p));
        Data{j} = Lstar+Z;
    end
    
    
    
    
    
    temp = randperm(n);
    train_ind = temp(1:floor(n*8/10));
    test_ind = temp(floor(n*8/10)+1:end);
    Y = zeros(p);
    for j = 1:n*8/10
        Y = Y+Data{j};
    end
    Y = Y/(n*8/10);
    
   
    
    kappa_bag = zeros(1,length(alpha_vec));  kappa_bag_upper=  zeros(length(alpha_vec),(p-rank)^2);
    for iter = 1:num_iter
        
        iter
        
        tic
        Data = cell(n,1);
        for j = 1:n
            Z = noise_degree*(Gauss_str(noise_ind)*Ustar*(diag(randn(p,1)))*Vstar(:,1:p)'+randn(p));
            Data{j} = Lstar+Z;
        end
        
        
        %% full data performance
        Y = zeros(p);
        for j = 1:n
            Y = Y+Data{j};
        end
        Y = Y/n;
        ymed = median(svd(Y));[U D V] = svd(Y);
        L = U(:,1:ranks)*D(1:ranks,1:ranks)*V(:,1:ranks)';
        rk = length(find(svd(L)>10^(-2)));
        
        temp_full = temp_full +(p-rank)^2-(trace(U(:,rk+1:end)*U(:,1+rk:end)'*perp_dirs_U)*trace(V(:,rk+1:end)*V(:,1+rk:end)'*perp_dirs_V));%*...
        
        
        
        
        %% bagging
        P_avg_U = zeros(p); P_avg_V = zeros(p);
        commutator_bag =  0;
        commutator_bag2 = 0;
        commutatro_bag_v2 = 0;
        for bag_iter = 1:num_bags
            
            t = randperm(n);
            Y = zeros(p);
            for j = 1:n/2
                Y = Y+Data{t(j)};
            end
            Y = Y/(n/2);
            [U D V] = svd(Y);
            L = U(:,1:ranks)*D(1:ranks,1:ranks)*V(:,1:ranks)';
            rk = length(find(svd(L)>10^(-2)));
            
            P_That_Uv{2*bag_iter-1} =  U(:,1:rk)*U(:,1:rk)';P_That_Vv{2*bag_iter-1} =  V(:,1:rk)*V(:,1:rk)';
            
            P_That_U = U(:,1:rk)*U(:,1:rk)'; P_That_V = V(:,1:rk)*V(:,1:rk)';
            P_avg_U = U(:,1:rk)*U(:,1:rk)' + P_avg_U;  P_avg_V = V(:,1:rk)*V(:,1:rk)' + P_avg_V;
            t1 = svd((eye(p)-P_That_U)*perp_dirs_U);
            t2 = svd((eye(p)-P_That_V)*perp_dirs_V);
            f = abs(sqrt(ones((p-rank)^2,1)-kron(t1(1:p-rank),t2(1:p-rank)).^2));
            commutatro_bag_v2 = commutatro_bag_v2+max(f.*sqrt(1-f.^2))^2;
            
            
            
            
            Y = zeros(p);
            for j = n/2+1:n
                Y = Y+Data{t(j)};
            end
            Y = Y/(n/2);
            [U D V] = svd(Y);
            L = U(:,1:ranks)*D(1:ranks,1:ranks)*V(:,1:ranks)';
            rk = length(find(svd(L)>10^(-2)));
            
            
            P_That_U = U(:,1:rk)*U(:,1:rk)'; P_That_V = V(:,1:rk)*V(:,1:rk)';
            P_avg_U = U(:,1:rk)*U(:,1:rk)' + P_avg_U;  P_avg_V = V(:,1:rk)*V(:,1:rk)' + P_avg_V;
            P_That_Uv{2*bag_iter} =  U(:,1:rk)*U(:,1:rk)';P_That_Vv{2*bag_iter} =  V(:,1:rk)*V(:,1:rk)';
            
            
        end
        
        
        
        % stability selection performance
        [U,~,~] = svd(P_avg_U);   [V,~,~] = svd(P_avg_V); Usel = U; Vsel = V;
        for i = 1:length(alpha_vec)
            alpha_thresh = alpha_vec(i);
            rk = length(find(min(svd(P_avg_U/(2*num_bags)),svd(P_avg_V/(2*num_bags)))>=alpha_thresh));
            rank_stability(i,iter) = 2*p*rk-rk^2;
            temp_stability(i) = temp_stability(i) +(p-rank_star)^2-(trace(U(:,rk+1:end)*U(:,1+rk:end)'*perp_dirs_U))*...
                (trace(V(:,rk+1:end)*V(:,1+rk:end)'*perp_dirs_V));
            stability_U{i} = U(:,1:rk)*U(:,1:rk)';  stability_V{i} = V(:,1:rk)*V(:,1:rk)';
            
        end
        
        rank_unique = unique(rank_stability(:,iter));
         rank_unique =  rank_unique(end:-1:1);
        for unique_ind = 1:length(rank_unique)
            unique_vect{unique_ind} = find(rank_stability(:,iter) == rank_unique(unique_ind));
            alpha_init(unique_ind) = alpha_vec(unique_vect{unique_ind}(1));
        end
        
        
        %% compute the commuator term:
        for alpha_iter = 1:length(alpha_vec)
            alpha_iter
            if ismember(alpha_vec(alpha_iter),alpha_init)
                
                commutator_bag = 0;
                for bag_iter = 1:num_bags
                    
                    k = 0;
                    Ustarj_ind = [];Vstarj_ind = [];
                        
                    
                     for i = 1:p-rank
                            perp_dirs_Uk = Ustar(:,rank+i)*Ustar(:,rank+i)';
                            term_bag11U(i) = trace( perp_dirs_Uk*(eye(p)-P_That_Uv{2*bag_iter-1})*(eye(p)-stability_U{alpha_iter}));
                            term_bag12U(i) = trace( (eye(p)-P_That_Uv{2*bag_iter-1})*perp_dirs_Uk*(eye(p)-P_That_Uv{2*bag_iter-1})*(eye(p)-stability_U{alpha_iter}));
                            term_bag21U(i) = trace( perp_dirs_Uk*(eye(p)-P_That_Uv{2*bag_iter})*(eye(p)-stability_U{alpha_iter}));
                            term_bag22U(i) = trace( (eye(p)-P_That_Uv{2*bag_iter})*perp_dirs_Uk*(eye(p)-P_That_Uv{2*bag_iter})*(eye(p)-stability_U{alpha_iter}));

                            
                     end
                    
                      for i = 1:p-rank
                            perp_dirs_Vk = Vstar(:,rank+i)*Vstar(:,rank+i)';
                            term_bag11V(i) = trace( perp_dirs_Vk*(eye(p)-P_That_Vv{2*bag_iter-1})*(eye(p)-stability_V{alpha_iter}));
                            term_bag12V(i) = trace( (eye(p)-P_That_Vv{2*bag_iter-1})*perp_dirs_Vk*(eye(p)-P_That_Vv{2*bag_iter-1})*(eye(p)-stability_V{alpha_iter}));
                            term_bag21V(i) = trace( perp_dirs_Vk*(eye(p)-P_That_Vv{2*bag_iter})*(eye(p)-stability_V{alpha_iter}));
                            term_bag22V(i) = trace( (eye(p)-P_That_Vv{2*bag_iter})*perp_dirs_Vk*(eye(p)-P_That_Vv{2*bag_iter})*(eye(p)-stability_V{alpha_iter}));

                      end
                    
                      temp1 = -kron( 2*term_bag11U,term_bag11V)+(kron(2*term_bag12U,term_bag12V));
                      temp2 = -kron( 2*term_bag21U,term_bag21V)+(kron(2*term_bag22U,term_bag22V));

                      
                       commutator_bag2 = commutator_bag2+sum(max(temp1',temp2'));
                       
                                                
                   
                    
                end
                
                kappa_bag(alpha_iter) = kappa_bag(alpha_iter)+commutator_bag2/(num_bags);
                
            else
                kappa_bag(alpha_iter) = kappa_bag(alpha_iter-1);
            end
            toc
        end
        
        
        
        %% half data performance
        t = randperm(length(train_ind));
        Y = zeros(p);
        for j = 1:n/2
            Y = Y+Data{t(j)};
        end
        Y = Y/(n/2);
        [U D V] = svd(Y);
        L = U(:,1:ranks)*D(1:ranks,1:ranks)*V(:,1:ranks)';
        rk = length(find(svd(L)>10^(-2)));
        P_That_U = U(:,1:rk)*U(:,1:rk)'; P_That_V = V(:,1:rk)*V(:,1:rk)';
        
        
        %% compute individual components of term F_1
        k = 1;
        
        for i = 1:rank_star+1:p
            for j=rank_star+1:p
                half_data_basis(k,noise_ind) = norm(P_That_U*Ustar(:,i)*Vstar(:,j)'+...
                    Ustar(:,i)*Vstar(:,j)'*P_That_V - P_That_U*Ustar(:,i)*Vstar(:,j)'*P_That_V,'fro');
                 
                k = k+1;
            end
        end
        temp_half_data_basis_dependent = temp_half_data_basis_dependent + squeeze(half_data_basis(:,noise_ind));
        
        
        %% compute the individual commutator among all directions
        k = 1;
        for i = 1:rank_star+1:p
            for j=1:rank_star+1:p
                perp_dirs_Ui = Ustar(:,i)*Ustar(:,i)';
                perp_dirs_Vj = Vstar(:,j)*Vstar(:,j)';
                
                commutator_indiv(k) = commutator_indiv(k)+ ...
                    sqrt(2*trace(perp_dirs_Ui*(eye(p)-P_That_U))*trace(perp_dirs_Vj*(eye(p)-P_That_V))-...
                    2*trace((eye(p)-P_That_U)*perp_dirs_Ui*(eye(p)-P_That_U)*perp_dirs_Ui)*...
                    trace((eye(p)-P_That_V)*perp_dirs_Vj*(eye(p)-P_That_V)*perp_dirs_Vj));
                k = k +1;
            end
        end
        
        %% compute overall comutator
        commutator_tot(iter) = 2*trace(perp_dirs_U*(eye(p)-P_That_U))*trace(perp_dirs_V*(eye(p)-P_That_V))-...
            2*trace((eye(p)-P_That_U)*perp_dirs_U*(eye(p)-P_That_U)*perp_dirs_U)*...
            trace((eye(p)-P_That_V)*perp_dirs_V*(eye(p)-P_That_V)*perp_dirs_V);
        
        temp_data_interactions_2 = temp_data_interactions_2+ kappa_bag/(2*num_bags);
        t1 = svd((eye(p)-P_That_U)*perp_dirs_U);
        t2 = svd((eye(p)-P_That_V)*perp_dirs_V);
        half_data_nobasis(1:(p-rank)^2,noise_ind) = half_data_nobasis(1:(p-rank)^2,noise_ind)+sort(abs(sqrt(ones((p-rank)^2,1)-kron(t1(1:p-rank),t2(1:p-rank)).^2)),'descend');
        toc
    end
    
    
    term_first = ((2*p*rank_const-rank_const^2)/p^2+min(commutator_indiv/iter))^2*p^2;
    term_second = 2*(1-alpha_vec).*mean(rank_stability,2);
    term_third =temp_data_interactions_2/iter;
    total_bound_2(:,config) = sum((temp_half_data_basis_dependent/iter).^2)+term_second+term_third';
    total_bound_1(:,config) = term_first+term_second+term_third';
    fd_stability(:,config) = temp_stability/iter;
    avg_discovery(:,config) = mean(rank_stability,2);
    fd_nonsubsampling(config) = temp_full/iter;
    commutator_tot_avg(config) = mean(commutator_tot);
end

close all;

for j = 1:length(Gauss_str)
    figure;
    
    plot(alpha_vec,fd_stability(:,j),'LineWidth',3)
    hold on;
    plot(alpha_vec,total_bound_2(:,j),'LineWidth',3)
    %plot(alpha_vec,total_bound_1(:,j),'LineWidth',3)
    plot(alpha_vec,avg_discovery(:,j),'LineWidth',2)
    hold on;
    plot(alpha_vec,fd_nonsubsampling(j)*ones(length(alpha_vec),1),'k--','LineWidth',3)
    hold on;
    set(gca,'FontSize',16,'FontWeight','bold')
    xlabel('\alpha')
    ylabel('FD')
    legend('W-S','Thm bound','Average discovery','N-S')
    set(gcf,'color','w');
    xlim([0.75,0.97])
end

