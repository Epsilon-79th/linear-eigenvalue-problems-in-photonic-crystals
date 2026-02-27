function run_timecmp(Ns,lattice_type,alpha)

if nargin==2
    alpha=[pi pi pi];
end

m_conv=15;

for i=1:length(Ns)
    run_timecmp_single(Ns(i),m_conv,lattice_type,alpha);
end

end

function run_timecmp_single(N,m_conv,lattice_type,alpha)

%% Addpath.

addpath('dielectric/');         % dielectric settings, savings.
addpath("discretization/");     % discrete scheme, FFT.
addpath("lobpcg/");             % eigensolver: lobpcg.

eigensolver_gpu=@PCs_linear_lobpcg_single_gpu;
eigensolver_cpu=@PCs_linear_lobpcg_single_cpu;

%% Basic settings: parameters.

a=2*pi*N^(2/3); 
k=1;
n=N*N*N;
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

%fprintf('len(M1)=%d.\n',length(M1));

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

%% Time comparison

[lam,~,iters_gpu,~]=eigensolver_gpu(D,B,Ms,INV,X0,m_conv,shift,tol/a);
fprintf('(Grid size N=%d): Iterations=%d, GPU runtime=%gs.\n',N,iters_gpu(1),iters_gpu(2));

for i=1:m_conv
     fprintf("i=%d, %9.5g.\n",i,a*sqrt(lam(i))/(2*pi))
end

[~,~,iters_cpu,~]=eigensolver_cpu(D,B,Ms,INV,X0,m_conv,shift,tol/a);
fprintf('(Grid size N=%d): Iterations=%d, CPU runtime=%gs.\n',N,iters_cpu(1),iters_cpu(2));

fprintf('\nAccelerating ratio (cpu_time/gpu_time)=%g.\n',iters_cpu(2)/iters_gpu(2));

%% Save data to output directory.

result_name=[lattice_type,'_',num2str(N)];

eval([result_name,'=struct("iter",iters_gpu(1),"gputime",iters_gpu(2),"cputime",iters_cpu(2),"ratio",iters_cpu(2)/iters_gpu(2));']);

if exist(['output/cmp_',lattice_type,'.mat'],'file')
    save(['output/cmp_',lattice_type,'.mat'],result_name,'-append');
else
    save(['output/cmp_',lattice_type,'.mat'],result_name);
end

%% Remove paths.

rmpath("dielectric/");         % dielectric settings, savings.
rmpath("discretization/");     % discrete scheme, FFT.
rmpath("lobpcg/");             % eigensolver: lobpcg.

end