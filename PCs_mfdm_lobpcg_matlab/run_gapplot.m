function run_gapplot(N,lattice_type,gap,indices)

%% How outputs are stored:

% For example, file 'bandgap_sc_curv.mat' contains several structs,
% each struct refers to a grid size .
% 

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
tol=1e-5;

bandgap_filename=['output/bandgap_',lattice_type,'.mat'];
gap_name=[lattice_type,'_',num2str(N)];

%% Set up the dielectric coefficients.

[M1,sym_points,CT]=dielectric_setup(N,lattice_type);
Ms=struct('inds',M1,'eps',1/eps);
clear M1 eps

%% Symmetry points in the lattice.

[~,n_pt]=size(sym_points); n_pt=n_pt-1;

if nargin==3
    indices=1:n_pt*gap;
end

if max(indices)>n_pt*gap || min(indices)<1
   error('Incompatible index range and gap size.\n'); 
end

Alpha=zeros(3,n_pt*gap);
for i=1:n_pt
    Alpha(:,i*gap)=sym_points(:,i+1);
    for j=1:gap-1
        Alpha(:,(i-1)*gap+j)=(j*sym_points(:,i+1)+(gap-j)*sym_points(:,i))/gap;
    end
end

%% Matrix blocks.

[D,Di]=mfd_fft_blocks(a,N,k,CT);

%% Preloading.

fprintf('\nrun_gapplot.m reminding: ');
if exist(bandgap_filename,'file')
    gap_names=who('-file',bandgap_filename);
    flag_exist=false;
    for i=1:length(gap_names)
        flag_exist=strcmp(gap_names{i},gap_name);
        if flag_exist
            break;
        end
    end
    
    if flag_exist
        fprintf('%s has a previous record and will be updated.\n',gap_name);
        load(bandgap_filename,gap_name);
        gap_rec=eval(gap_name);
    else
        fprintf('N=%d is a new grid size for lattice type "%s".\n',N,lattice_type);
        gap_rec=struct('eigen',zeros(n_pt*gap,m_conv),'iter',zeros(n_pt*gap,2));
        eval([gap_name,'=gap_rec;']);
    end
else
    fprintf('%s is a new lattice type.\n',lattice_type);
    gap_rec=struct('eigen',zeros(n_pt*gap,m_conv),'iter',zeros(n_pt*gap,2));
    eval([gap_name,'=gap_rec;']);
    save(bandgap_filename,gap_name);
end
fprintf('\n');

%% Compute eigenvalues of all gap points.

X0=rand(3*n,m_conv+max(10,m_conv));
for i=1:length(indices)

    wait(gpuDevice)
    t_mat_flag=tic;
    alpha=Alpha(:,indices(i))/a;

    norm_alpha=norm(alpha);
    if norm_alpha>pi/5/a
        m=round(m_conv*1.35);
        pnt=2*N*a;
        shift=0;
    elseif norm_alpha==0
        pnt=2*N*a;
        m=m_conv+5;
        shift=1/(4*pi)/N;
    else
        m=m_conv+10;
        pnt=2*N/(norm_alpha);
        shift=a*norm_alpha/N;
    end
    
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

    clear("d11");clear("d22");clear("d33");
    clear("d12");clear("d13");clear("d23");
    
    wait(gpuDevice)
    t_mat=toc(t_mat_flag);
    fprintf('time for assembling matrix: %gs.\n',t_mat);
    
    % Eigensolver.
    [lambda_pnt,lambda_re,iters_,X0(:,1:m)]=eigensolver(D,B,Ms,INV,X0(:,1:m),m_conv,shift,tol/a);
    X0=gather(X0);

    gap_rec.eigen(indices(i),:)=a*sqrt(lambda_re)/(2*pi);
    gap_rec.iter(indices(i),:)=iters_;

    lambda_devia=a*abs(sqrt(lambda_pnt)-sqrt(lambda_re))/(2*pi);
    
    for j=1:m_conv
        fprintf('lambda_pnt=%g, lambda_re=%g, devia=%g, w/(2pi)=%g.\n',lambda_pnt(j),lambda_re(j),...
            lambda_devia(j),gap_rec.eigen(indices(i),j));
    end
    
    fprintf('\n\ni=(%d/%d) completes, iterations=%d, time=%gs.\n\n',indices(i),n_pt*gap,iters_(1),iters_(2));
    
    eval([gap_name,'=gap_rec;']);
    save(bandgap_filename,gap_name,'-append');

end

%% Remove paths.

rmpath("dielectric/");         % dielectric constant.
rmpath("discretization/");     % discrete scheme, FFT.
rmpath("lobpcg/");             % eigensolver: lobpcg.

end
