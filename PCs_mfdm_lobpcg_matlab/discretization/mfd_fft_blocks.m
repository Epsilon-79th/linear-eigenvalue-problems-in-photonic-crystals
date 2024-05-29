function [D,Di]=mfd_fft_blocks(a,N,k,CT)

%a: lattice constant;
%N: grid size;
%k: accuracy;
%alpha: shift in Brillouin zone.
%eps: permittivity constant.

h=a/N;

%% MFDM discretization 

load Finite_Difference_Stencils fd_stencil;
fd_stencil_0=fd_stencil(:,1:8);
fd_stencil_1=fd_stencil(:,9:16);

%% FFT eigenvalues
tic;

%global INV_D;

fd=fd_stencil_1(k,1:2*k)/h;
fdi=fd_stencil_0(k,1:2*k);

I_N=ones(N,1);I_N2=ones(N^2,1);

D0=find_eig(fd(k:end),fd(k-1:-1:1),N);
D01=kron_diag(I_N2,D0);
D02=kron_diag(I_N,kron_diag(D0,I_N));
D03=kron_diag(D0,I_N2);
Di=find_eig(fdi(k:end),fdi(k-1:-1:1),N);

D=[CT(1,1)*D01+CT(1,2)*D02+CT(1,3)*D03;...
   CT(2,1)*D01+CT(2,2)*D02+CT(2,3)*D03;...
   CT(3,1)*D01+CT(3,2)*D02+CT(3,3)*D03];
Di=[kron_diag(I_N2,Di);kron_diag(I_N,kron_diag(Di,I_N));kron_diag(Di,I_N2)];

t0=toc;
fprintf('time for computing FFT-diag matrix: %gs.\n',t0);

end
