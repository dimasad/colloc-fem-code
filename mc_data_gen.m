%%%% Data generation for Monte-Carlo empirical study %%%%
%% Setup random number generator and define parameters
rng(0)

nx = 5;
nu = 3;
ny = 3;
nw = nx;

N = 500;

std_u = 1;
std_w = 0.05;
std_v = 0.2;

%% Create system
rng_settings = rng;

sys = drss(nx, ny, nu + nw);
sys.d(:, nu+1:end) = 0;

Q = eye(nw) * std_w^2;
R = eye(ny) * std_v^2;

[~, L, Pp] = kalman(sys, Q, R);

u = std_u * randn(N, nu);
w = std_w * randn(N, nw);
v = std_v * randn(N, ny);

y = lsim(sys, [u, w]) + v;

%% Save
save -ascii /tmp/u.txt u
save -ascii /tmp/y.txt y

%% 
d = iddata(y, u, 1);
syse = n4sid(d, nx);
[syseb, gram, T] = balreal(syse);

%%
A = syseb.a;
B = syseb.b;
C = syseb.c;
D = syseb.d;
K = T * syse.k;

xpred = zeros(N, nx);
epred = zeros(N, ny);
for i=1:N-1
    ui = u(i, :)';
    xi = xpred(i, :)';
    yi = y(i, :)';
    ei = yi - C*xi - D*ui;
    xnext = A*xi + B*ui + K*ei;
    
    xpred(i+1, :) = xnext';
    epred(i, :) = ei';
end
epred(N,:) = y(N, :) - xpred(N,:) * C' - u(N,:)*D';

Rp = cov(epred);
sRp = chol(Rp)';
isRp = inv(sRp);

%% 
save -ascii /tmp/a.txt A
save -ascii /tmp/b.txt B
save -ascii /tmp/c.txt C
save -ascii /tmp/d.txt D
save -ascii /tmp/k.txt K
save -ascii /tmp/xpred.txt xpred
save -ascii /tmp/epred.txt epred
save -ascii /tmp/gram.txt gram
save -ascii /tmp/isRp.txt isRp

%%
