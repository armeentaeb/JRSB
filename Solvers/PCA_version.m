function[X,runhist] = PCA_version(Sigma,p,q,rho,beta,gamma2)
HOME = '/Users/armeentaeb/Dropbox/JRSB_Data_CODE/Solvers/logdetppa-0'; 
   addpath(strcat(HOME,'/solver/'))
   addpath(strcat(HOME,'/solver/mexfun'))
   addpath(strcat(HOME,'/util/'))
   
    % here is the start of it    
      m = (p+q)*(p+q+1)/2 - (q)*(q+1)/2;
      n = p+q;
      nyy = p;
      [M1 M2] = ObtainMatrices_nucNorm(p,q); 
      blk{1,1} = 's'; blk{1,2} = (p+q);
      C{1,1} = Sigma;
      M1 = [M1];
      At{1,1} = sparse(M1);
      
      %% this is the Lyy block

      blk{2,1} = 's'; blk{2,2} = p;
      
      for k = 1:p*(p+1)/2
          Temp2(k,:) = [k k 1];
      end
  
      Temp2(p*(p+1)/2+1,:) = [p*(p+1)/2 m 0];
    
      Atmp = spconvert(Temp2);
      At{2,1}  = Atmp; 
      C{2,1}   = beta*speye(p,p); 
      
      
      
         %% this is the delta block
       
      blk{3,1} = 's'; blk{3,2} = p+q;
       M2 = [M2];
      At{3,1}  = sparse(M2); 
      C{3,1}   = gamma2*speye(n,n); 
      
      %% This is the diagonal component
      
     blk{4,1} = 'l'; blk{4,2} = nyy*(nyy+1)/2;  
      startp = 1;
      ind = 2;
      for t = 1:p*(p+1)/2
          
            if t == startp
                Temp4(t,:) = [startp t 1];
                startp = startp + ind;
                ind = ind+1;
            else
                Temp4(t,:) = [startp t 0];
            end
         
      end
      Temp4(t+1,:) = [nyy*(nyy+1)/2  m 0];

      
      
      At{4,1} = [-spconvert(Temp4)];
     
      C{4,1} = zeros(nyy*(nyy+1)/2,1);
      
      
      
      b = zeros(m,1);
      runPPA = 1; 
      if (runPPA)
         OPTIONS.smoothing  = 1;
         OPTIONS.scale_data = 0; %% or 2;
         OPTIONS.plotyes    = 0; 
         OPTIONS.tol        = 1e-6;
         mu = [1; 0; 0;0];
         [obj,X,y,Z,runhist] = logdetPPA(blk,At,C,b,mu,OPTIONS);
         
      end

  % How to check to see how accurate our model is 