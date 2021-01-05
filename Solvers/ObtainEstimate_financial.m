function [F,S_Y] = ObtainEstimate_financial(TrainTestData,method,p,q,rho,beta,gamma)
HOME = '~/logdetppa-0'; 


% load Training and testing data
TrainY = TrainTestData{1};
TestY = TrainTestData{2};
TrainX = TrainTestData{3}(:,:);
TestX = TrainTestData{4}(:,:);

% Compute scaled smaple covariance
%SigmaTrain = cov([TrainY' TrainX']);
SigmaTrain = cov([TrainY']);

Dtrans = diag(sqrt(1./diag(SigmaTrain)));
%Dtrans = eye(p+q);
SigmaTrain = SigmaTrain;
DtransY = Dtrans(1:p,1:p);


global LVGM;
global Composite_LVGM;
global FM;
global Composite_FM;
global GM;


switch method
    case LVGM
   [F,runhist] = LnuclNorm_version(SigmaTrain(1:p,1:p),p,0,rho,beta,gamma);
   
    S_Y = F{1}(1:p,1:p) + F{2};
    Spars = length(find(abs(S_Y) < 10^(-4)))/p^2;
    numHidden = length(find(svd(F{2})>10^(-4)));
    ModelComplex = (1-Spars)*p^2/2 + p/2+ numHidden*p-numHidden*(numHidden-1)/2;
    BestLinearEstimator = zeros(p,q);
    [U D V] = svd(F{1}(1:p,1:p)^(-1) - S_Y^(-1));
    disp(sprintf('Total number of parameters is %d', ModelComplex))
    disp(sprintf('total number of edges is %d', (1-Spars)*p^2/2 - p/2))
    disp(sprintf('Number of hidden variables is %d', numHidden))
    
        case GM
   [F,runhist] = LnuclNorm_version(SigmaTrain(1:p,1:p),p,0,rho,beta,gamma);
   
    S_Y = F{1}(1:p,1:p);
    Spars = length(find(abs(S_Y) < 10^(-3)))/p^2;
    ModelComplex = (1-Spars)*p^2/2 + p/2;
    BestLinearEstimator = zeros(p,q);
    disp(sprintf('Total number of parameters is %d', ModelComplex))
    disp(sprintf('total number of edges is %d', (1-Spars)*p^2/2 - p/2))
 
      
   
    case Composite_LVGM
    
    [F,runhist] = LnuclNorm_version(SigmaTrain,p,q,rho,beta,gamma);
    S_Y = F{1}(1:p,1:p) + F{2};
    Spars = length(find(abs(S_Y) < 10^(-4)))/p^2;
    numHidden = length(find(svd(F{2}) > 10^(-4)));
    SDRdim = length(find(svd(F{1}(1:p,p+1:end)) > 10^(-4)));
    ModelComplex = SDRdim*(p+q)-SDRdim^2 + (1-Spars)*p^2/2 + p/2+ numHidden*p-numHidden*(numHidden-1)/2;
    BestLinearEstimator = -(F{1}(1:p,1:p))^(-1)*(F{1}(1:p,p+1:end));    
    [Up D Vp] = svd(BestLinearEstimator);
    [Upp D Vpp] = svd(F{1}(1:p,1:p)^(-1)-S_Y^(-1));
    
     disp(sprintf('Total number of parameters is %d', ModelComplex))
    disp(sprintf('total number of edges is %d', (1-Spars)*p^2/2 - p/2))
    disp(sprintf('Number of hidden variables is %d', numHidden))
    disp(sprintf('SDR dimension is %d', SDRdim))
      
    case FM
        
        [F,runhist] = PCA_version(SigmaTrain(1:p,1:p),p,0,rho,beta,gamma);
       numHidden = length(find(svd(F{2}) > 10^(-5)));
        ModelComplex =  numHidden*p-numHidden*(numHidden-1)/2;
        S_Y = F{1}(1:p,1:p) + F{2};
        BestLinearEstimator = zeros(p,q);
       %disp(sprintf('Total number of parameters is %d', ModelComplex))
       % disp(sprintf('Number of hidden variables is %d', numHidden))

    
    case Composite_FM
        [F,runhist] = PCA_version(SigmaTrain,p,q,rho,beta,gamma);
        numHidden = length(find(svd(F{2}) > 10^(-5)));
        S_Y = F{1}(1:p,1:p) + F{2};
        SDRdim = length(find(svd(F{1}(1:p,p+1:end)) > 10^(-5)));
        ModelComplex = numHidden*p-numHidden*(numHidden-1)/2 + SDRdim*(p+q)-SDRdim^2;
        BestLinearEstimator = zeros(p,q);       
        disp(sprintf('Total number of parameters is %d', ModelComplex))
        disp(sprintf('Number of hidden variables is %d', numHidden))
        disp(sprintf('SDR dimension is %d', SDRdim))
 
end


%  %% Predictive performance
%  % compute the mean
%  TrainY = TrainY';
%  TrainX = TrainX';
%  TestY = TestY';
%  TestX = TestX';
% 
% 
%     for i = 1:p
%         ind = find(~isnan(TrainY(:,i)));
%          avgY(1,i) = mean(TrainY(ind,i),1);
%     end
%     
% 
%     sizeNonan = 1;
%     for j = 1:size(TestY,1)
%         if sum(isnan(TestY(j,:))) == 0 & sum(isnan(TestX(j,:))) == 0
%             TestYnonan(sizeNonan,1:size(TestY,2)) = TestY(j,:);
%             TestXnonan(sizeNonan,1:size(TestX,2)) = TestX(j,:);
%               sizeNonan = sizeNonan + 1;
%         end
%     end
% 
% % 
%     NumTest = size(TestYnonan,1);
%     ConditionalOutofSample = DtransY*(TestYnonan' - repmat(avgY',1,NumTest));%- BestLinearEstimator*DtransX*(TestXnonan' - repmat(avgX',1, NumTest));
%     LogLikelihood = 0;
%     for i = 1:NumTest
%          l = ConditionalOutofSample(:,i);
%          Sig = F{1}^(-1);
%          MarginalDist = Sig(1:p,1:p)^(-1);
%          temp = -p/2*log(2*pi)+1/2*sum(log(eig(MarginalDist)))-1/2*l'*MarginalDist*l;
%          LogLikelihood = LogLikelihood + temp;
%     end
% % 
% 
% 
% 
%   %% Compute in sample performance
%   sizeNonan = 1;
%     for j = 1:size(TrainY,1)
%         if sum(isnan(TrainY(j,:))) == 0 & sum(isnan(TrainX(j,:))) == 0
%             TrainYnonan(sizeNonan,1:size(TrainY,2)) = TrainY(j,:);
%             TrainXnonan(sizeNonan,1:size(TrainX,2)) = TrainX(j,:);
%              sizeNonan = sizeNonan + 1;
%         end
%     end
% 
% 
% %     
%     NumTrain = size(TrainXnonan,1);
%     L = [TrainY]'-repmat([avgY'],1,NumTrain);
%     insample = -(p+q)/2*log(2*pi)+1/2*sum(log(eig(F{1})))-1/2*trace(Dtrans*L*L'*Dtrans'*F{1})/NumTrain;
%     
%     ConditionalOutofSample = Dtrans*(TrainYnonan' - repmat(avgY',1,NumTrain));%- BestLinearEstimator*DtransX*(TrainXnonan' - repmat(avgX',1, NumTrain));
%     LogLikelihoodtrain = 0;
%     for i = 1:NumTrain
%          l = ConditionalOutofSample(:,i);
%          Sig = F{1}^(-1);
%          MarginalDist = F{1};
%          temp = -p/2*log(2*pi)+1/2*sum(log(eig(MarginalDist)))-1/2*l'*MarginalDist*l;
%          LogLikelihoodtrain = LogLikelihoodtrain + temp;
%     end
%     %LogLikelihoodtrain = 0;
% 
%   insample =   LogLikelihoodtrain/NumTrain;
%   outsample = LogLikelihood/NumTest;
% %                                                                                                                                                
% %  disp(sprintf('log likelihood on training samples is %d', LogLikelihoodtrain/NumTrain))   
% %  disp(sprintf('log likelihood on future samples is %d', LogLikelihood/NumTest))
