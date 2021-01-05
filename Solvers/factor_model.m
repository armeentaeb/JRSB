function [F,rank_regular,validation] = factor_model(Y_train,Y_test,lambda)

global FM
FM = 1;
TrainTest{1} = Y_train';
TrainTest{2} = Y_test';
TrainTest{3} = zeros(size(Y_train'));
TrainTest{4} = zeros(size(Y_test'));
p = size(Y_train,2);

[F,~] = ObtainEstimate_financial(TrainTest,FM,p,0,0,lambda,0);
rank_regular = sum(svd(F{2})>10^(-3));
%validation = test_performance(F{1},Y_train,Y_test);
validation = 10;


end

