function run_tolcmp(N,lattice_type,tols,alpha)

if nargin==3
    alpha=[pi pi pi];
end

%% Addpath.

addpath('dielectric/');         % dielectric settings, savings.
addpath("discretization/");     % discrete scheme, FFT.
addpath("lobpcg/");             % eigensolver: lobpcg.

eigensolver=@PCs_linear_lobpcg_single_gpu;

%% Parameters.

a=2*pi*N^(2/3);
k=1;
m_conv=15;
m=20;
n=N*N*N;
pnt=2*N*a;
eps=13;
shift=0;

X0=rand(3*n,m);

%% Set up the dielectric coefficients.

[M1,~,CT]=dielectric_setup(N,lattice_type);
Ms=struct('inds',M1,'eps',1/eps);
clear M1 eps

%% Matrix blocks.

[D,Di]=mfd_fft_blocks(a,N,k,CT);
D(1:n)=D(1:n)+1i*alpha(1)*Di(1:n);
D(n+1:2*n)=D(n+1:2*n)+1i*alpha(2)*Di(n+1:2*n);
D(2*n+1:end)=D(2*n+1:end)+1i*alpha(3)*Di(2*n+1:end);

d11=D(1:n).*conj(D(1:n));
d22=D(n+1:2*n).*conj(D(n+1:2*n));
d33=D(2*n+1:end).*conj(D(2*n+1:end));
d12=D(n+1:2*n).*conj(D(1:n));
d13=D(2*n+1:end).*conj(D(1:n));
d23=D(2*n+1:end).*conj(D(n+1:2*n));

B=[d11;d22;d33;d12;d13;d23]*pnt;

% FFT inverse of AA'+pnt B'B.
INV=inverse_3_times_3_blocks([pnt*d11+d22+d33+shift;...
    d11+pnt*d22+d33+shift;d11+d22+pnt*d33+shift],...
    (pnt-1)*[d12;d13;d23]);

clear d11 d22 d33 d12 d13 d23

n_tols=length(tols);
lambda_pnt=zeros(n_tols,m_conv);
lambda_re=zeros(n_tols,m_conv);
iters=zeros(n_tols,2);

for i=1:n_tols    
    [lambda_pnt(i,:),lambda_re(i,:),iters(i,:),~]=eigensolver(D,B,Ms,INV,X0,m_conv,shift,tols(i)/a);
    lambda_pnt(i,:)=a*sqrt(lambda_pnt(i,:))/(2*pi);
    lambda_re(i,:)=a*sqrt(lambda_re(i,:))/(2*pi);
    fprintf('\ntol=%.1e is computed.\n\n',tols(i));
end

std_pnt=std(lambda_pnt);
std_re=std(lambda_re);

fprintf('Tolerance: \n');
for i=1:n_tols
    fprintf('%.1e,\titer=%d,\ttime=%gs.\n',tols(i),iters(i,1),iters(i,2));
end

fprintf('\nStandard deviation of lambdas_pnt, lambdas_re:\n');
for i=1:m_conv
    fprintf('i=%d,\t%6.3e,\t%6.3e.\n',i,std_pnt(i),std_re(i));
end

%% Remove paths.

rmpath("dielectric/");         % dielectric settings, savings.
rmpath("discretization/");     % discrete scheme, FFT.
rmpath("lobpcg/");             % eigensolver: lobpcg.

end