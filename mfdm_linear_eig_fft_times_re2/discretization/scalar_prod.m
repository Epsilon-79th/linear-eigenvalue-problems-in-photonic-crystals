function X=scalar_prod(X,M_ind,eps)

if nargin==3
    X(M_ind,:)=X(M_ind,:)*eps;
else
    X(M_ind.inds,:)=X(M_ind.inds,:)*M_ind.eps;
end

end