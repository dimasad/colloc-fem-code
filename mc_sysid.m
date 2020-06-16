%% Setup analysis

data_dir = strcat('data', filesep, 'mc_experim');
config = load([data_dir, filesep, 'config']);

%% Loop through all experiments

for e = dir(strcat(data_dir, filesep, 'exp*.mat'))'
    %% Load data
    data = load([e.folder, filesep, e.name]);
    d = iddata(data.y, data.u);
    dv = d(1:end/2);
    de = d(end/2+1:end);
    
    %% Estimate and get balanced realization
    s1 = n4sid(de, nx);
    s2 = ssest(de, nx);
    [syseb, gram, T] = balreal(s1);
    
    A = syseb.a;
    B = syseb.b;
    C = syseb.c;
    D = syseb.d;
    L = T * s1.k;

    %% Run predictor
    xpred = zeros(N, nx);
    epred = zeros(N, ny);
    for i=1:N-1
        ui = u(i, :)';
        xi = xpred(i, :)';
        yi = y(i, :)';
        ei = yi - C*xi - D*ui;
        xnext = A*xi + B*ui + L*ei;
        
        xpred(i+1, :) = xnext';
        epred(i, :) = ei';
    end
    epred(N,:) = y(N, :) - xpred(N,:) * C' - u(N,:)*D';
    
    %% Get innovation covariance and normalized innovations
    Rp = cov(epred);
    sRp = chol(Rp)';
    isRp = inv(sRp);
    
    en = (sRp \ epred')';
    %% Save
end
