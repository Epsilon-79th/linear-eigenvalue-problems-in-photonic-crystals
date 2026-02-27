function run_scalcmp(N,lattice_type,as,alpha)

if nargin==3
    alpha=[pi pi pi];
end

%% Add paths.

addpath("dielectric/");         % dielectric constant.
addpath("discretization/");     % discrete scheme, matrix.
addpath("lobpcg/");             % eigensolver: lobpcg.

eigensolver=@PCs_linear_lobpcg_single_gpu;

%% Basic settings: parameters.
k=1;
n=N*N*N;
m_conv=15;
eps=13;
a0=2*pi;

alpha=alpha/a0;
X0=rand(3*n,round(m_conv*1.4));
tol=1e-5;

%% Set up the dielectric coefficients.

[M1,~,CT]=dielectric_setup(N,lattice_type);
Ms=struct('inds',M1,'eps',1/eps);
clear M1 eps

%% Matrix blocks.

[D,Di]=mfd_fft_blocks(a0,N,k,CT);
D(1:n)=D(1:n)+1i*alpha(1)*Di(1:n);
D(n+1:2*n)=D(n+1:2*n)+1i*alpha(2)*Di(n+1:2*n);
D(2*n+1:end)=D(2*n+1:end)+1i*alpha(3)*Di(2*n+1:end);

d11=D(1:n).*conj(D(1:n));
d22=D(n+1:2*n).*conj(D(n+1:2*n));
d33=D(2*n+1:end).*conj(D(2*n+1:end));
d12=D(n+1:2*n).*conj(D(1:n));
d13=D(2*n+1:end).*conj(D(1:n));
d23=D(2*n+1:end).*conj(D(n+1:2*n));

n_as=length(as);
lambda_pnt=zeros(n_as,m_conv);
lambda_re=zeros(n_as,m_conv);
iters=zeros(n_as,2);

for i=1:n_as
    a=N^as(i);
    pnt=2*N*a;

    % FFT inverse of AA'+pnt B'B.
    INV=inverse_3_times_3_blocks([pnt*d11+d22+d33;...
            d11+pnt*d22+d33;d11+d22+pnt*d33]/(a*a),...
            (pnt-1)*[d12;d13;d23]/(a*a));

    [lambda_pnt(i,:),lambda_re(i,:),iters(i,:),~]=eigensolver(D/a,...
     pnt*[d11;d22;d33;d12;d13;d23]/(a*a),Ms,INV,X0,m_conv,0,tol/a);
    lambda_pnt(i,:)=a*sqrt(lambda_pnt(i,:));
    lambda_re(i,:)=a*sqrt(lambda_re(i,:));

    fprintf('\nscaling=N^%.2f is computed.\n',as(i));
    for j=1:m_conv
        fprintf('i=%d, %g\n',j,lambda_pnt(i,j));
    end
    fprintf('\n');
end

std_pnt=std(lambda_pnt);
std_re=std(lambda_re);

fprintf('Scaling: \n');
for i=1:n_as
    fprintf('N^%.2f,\titer=%d,\ttime=%gs.\n',as(i),iters(i,1),iters(i,2));
end

fprintf('\nStandard deviation of lambdas_pnt, lambdas_re:\n');
for i=1:m_conv
    fprintf('i=%d,\t%6.3e,\t%6.3e.\n',i,std_pnt(i),std_re(i));
end

%% Remove paths.

rmpath("dielectric/");         % dielectric constant.
rmpath("discretization/");     % discrete scheme, FFT.
rmpath("lobpcg/");             % eigensolver: lobpcg.

end