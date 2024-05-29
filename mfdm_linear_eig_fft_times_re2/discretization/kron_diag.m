function C=kron_diag(A,B)

%% Kronecker product of diagonal matrix A,B.


%A=(a_1,\cdots,a_n).
%B=(b_1,\cdots,b_m).
%C=diag(kron(diag(A),diag(B))).

A=reshape(A,1,[]);
B=reshape(B,[],1);

C=reshape(B*A,[],1);

end