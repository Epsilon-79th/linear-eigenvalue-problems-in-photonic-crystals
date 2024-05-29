function run_1p(N,lattice_type,alpha)

% Compute eigenvalues of a single lattice vector.

%% Add paths.

addpath("dielectric/");         % dielectric constant.
addpath("discretization/");     % discrete scheme, matrix.
addpath("lobpcg/");             % eigensolver: lobpcg.

eigensolver=@PCs_linear_lobpcg_single_gpu;

%% Basic settings: parameters.
a=2*pi*N^(2/3); 
k=1;
n=N*N*N;
m_conv=15;
eps=13;
alpha=alpha/a;

X0=rand(3*n,round(m_conv*1.4));
pnt=2*N;
shift=0;
tol=1e-5;

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

%% Output.
[lambda_pnt,lambda_re,iters,~]=eigensolver(D,B,Ms,INV,X0,m_conv,shift,tol/a);
lambda_pnt=a*sqrt(lambda_pnt)/(2*pi);
lambda_re=a*sqrt(lambda_re)/(2*pi);

fprintf('Convergence takes %d iterations, %gs elapsed.\n',iters(1),iters(2));

fprintf('Lattice type: %s, alpha=[%g,%g,%g]pi, grid size N=%d.\n',lattice_type,...
        a*alpha(1)/pi,a*alpha(2)/pi,a*alpha(3)/pi,N);
fprintf('First %d eigenvalues are (pnt, recomp, devia):\n',N);
for i=1:m_conv
    fprintf('i=%d, %g, %g, %g.\n',i,lambda_pnt(i),lambda_re(i),abs(lambda_pnt(i)-lambda_re(i)));
end

%% Remove paths.

rmpath("dielectric/");         % dielectric constant.
rmpath("discretization/");     % discrete scheme, FFT.
rmpath("lobpcg/");             % eigensolver: lobpcg.

end