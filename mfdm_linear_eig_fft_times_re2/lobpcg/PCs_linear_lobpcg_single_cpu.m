function [lambda_pnt,lambda_re,iter_,X]=PCs_linear_lobpcg_single_cpu(Ds,Bs,Ms,INV,X0,m_conv,shift,tol_lobpcg)

%% Input.
% Ds: FFT diagonalization of curl.
% Bs: FFT diagonalization of div, the penalty term.
% Ms: diagonal matrix of dielectric coefficient.
% INV_D: FFT inverse of AA'+pnt B'B+shift.
% X0: initial guess.
% m_conv: number of desired eigenpairs.
% shift: (to ensure invertibility).
% tol_lobpcg: tolerance.

%% Output.
% lambda_0: eigenvalues of penalty scheme.
% lambda_re: recomputing eigenvalues.
% iter_: iterations and time.
% X: eigenvectors.

%% LOBPCG: initialization

%Maximum lobpcg iterations
iter_max=1e3;

[~,n_col]=size(X0);
X=X0;

P=[];
iter=0;

%wait(gpuDevice);
t_total_flag=tic;

%wait(gpuDevice);
t_init_flag=tic;
HX=A_fft(scalar_prod(A_fft(X,-conj(Ds)),Ms),Ds)+H_fft_upper(X,Bs)+shift*X; 

lambda=eig(X'*HX,X'*X);
lambda=sort(real(lambda));

%wait(gpuDevice);
t_init=toc(t_init_flag);
fprintf('time for lobpcg initialization: %gs.\n',t_init);

%% LOBPCG: main loop.

%mat_conv=sparse(1:m_conv,1:m_conv,ones(1,m_conv),n_col,m_conv);
while iter<iter_max
    
    %wait(gpuDevice)
    t_single_iter_flag=tic;
    iter=iter+1;
    
    %wait(gpuDevice)
    t_else_flag=tic;

    R=HX-X*diag(lambda);

    %wait(gpuDevice)
    t_else=toc(t_else_flag);

    %wait(gpuDevice)
    t_norm_flag=tic;
    X_tmp=X'*X; 
    X_norm=sqrt(real(trace(X_tmp)));
    X_norms=sqrt(real(diag(X_tmp)));
    
    %compute residual and record it.
    
    R_tmp=R'*R;
    R_norm=sqrt(real(trace(R_tmp)));
    R_norms=sqrt(real(diag(R_tmp)));    

    %wait(gpuDevice)
    t_norm=toc(t_norm_flag);

    clear('X_tmp');clear('R_tmp');    
    
    %index of convergent components
    %ind_conv=find(abs(lambda(1:n_col)-lambda_pre(1:n_col))<=tol_lobpcg);
    ind_conv=find(R_norms<=tol_lobpcg*X_norms);
    
    %index of those still require iteration.
    %ind_act=find(abs(lambda(1:n_col)-lambda_pre(1:n_col))>eps);
    ind_act=find(R_norms>tol_lobpcg*X_norms);
    
    n_conv=length(ind_conv);
    n_act=length(ind_act);
    %n_act=n_col;
    
    if (n_conv>=m_conv) %|| R_norm_conv<tol_lobpcg*0.75
        break;
    end

    %
    %preconditioning.
    
    %wait(gpuDevice)
    t_precond_flag=tic;
    
    %R=R-R*R_tmp;
    W=H_fft_upper(R(:,ind_act),INV);
    HW=A_fft(scalar_prod(A_fft(W,-conj(Ds)),Ms),Ds)+H_fft_upper(W,Bs)+shift*W;

    %wait(gpuDevice)
    t_fft=toc(t_precond_flag);

    %wait(gpuDevice)
    t_else_flag=tic;
    
    if (iter>1)
        P=P(:,ind_act);
        HP=HP(:,ind_act);

        %T=[X,W,P]'*[HX,HW,HP];
        %G=[X,W,P]'*[X,W,P];
        T=[X'*HX,X'*HW,X'*HP;W'*HX,W'*HW,W'*HP;P'*HX,P'*HW,P'*HP];
        G=[X'*X,X'*W,X'*P;W'*X,W'*W,W'*P;P'*X,P'*W,P'*P];
        %[T,G]=ELSE_Together(X,HX,W,HW,P,HP);
    else
        
        %T=[X,W]'*[HX,HW];
        %G=[X,W]'*[X,W];
        T=[X'*HX,X'*HW;W'*HX,W'*HW];
        G=[X'*X,X'*W;W'*X,W'*W];
        %[T,G]=ELSE_Together(X,HX,W,HW);
    end
    
    T=(T+T')/2;G=(G+G')/2;
    
    %wait(gpuDevice)
    t_else=t_else+toc(t_else_flag);

    %wait(gpuDevice)
    t_small_eig_flag=tic;
    
    %
    %T=gather(T); G=gather(G);
    [S,D]=eig(T,G,'chol');
    %}
    
    [lambda,ind]=sort(real(diag(D)));
    lambda=lambda(1:n_col);
    U=S(:,ind(1:n_col));
    
    %wait(gpuDevice)
    t_small_eig=toc(t_small_eig_flag);
    %}
    
    %wait(gpuDevice)
    t_else_flag=tic;

    U_X=U(1:n_col,:);
    U_W=U(n_col+1:n_col+n_act,:);
    if (iter>1)
        U_P=U(n_col+n_act+1:n_col+2*n_act,:);
    end

    if (iter>1)
        P=W*U_W+P*U_P;
        HP=HW*U_W+HP*U_P;
        %MP=MW*U_W+MP*U_P;
    else
        P=W*U_W;
        HP=HW*U_W;
        %MP=MW*U_W;
    end
    
    X=X*U_X+P;
    HX=HX*U_X+HP;
    %MX=MX*U_X+MP;
    
    %wait(gpuDevice)
    t_else=t_else+toc(t_else_flag);
    
    %wait(gpuDevice)
    t_single_iter=toc(t_single_iter_flag);

    fprintf('iter=%d, time=%.3fs, eig=%.3fs, fft=%.3fs, norms=%.3fs, else=%.3fs, res=%.3g, n_act=%d.\n',...
        iter,t_single_iter,t_small_eig,t_fft,t_norm,t_else,R_norm/X_norm,n_act);

end

%% Data arrangement

lambda_pnt=real(lambda(1:m_conv)-shift);

%wait(gpuDevice);

if iter==iter_max
    error('Max iteration times (1e3) reached!.\n');
else
    fprintf('Iteration times=%g.\n',iter);
end
%fprintf('Maximum condition number occured in the loop: %g.\n',cond_max);

%{
if nargin==4
    residual_record=residual_record(1:iter);
end
%}

X_=X(:,1:m_conv);

t_post_flag=tic;
AMAX=X_'*A_fft(scalar_prod(A_fft(X_,-conj(Ds)),Ms),Ds);
lambda_re=real(diag(AMAX)./diag(X_'*X_));
t_post=toc(t_post_flag);
fprintf('time for recomputing=%gs.\n',t_post);

%wait(gpuDevice);
t_total=toc(t_total_flag);
iter_=[iter,t_total];

end