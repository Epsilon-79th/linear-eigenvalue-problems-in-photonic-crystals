function val=norm_gpu(R,mass)

%Usage: Compute the Frobenius norm of a multivector.
%Input: R multivector N*m gpu-matrix. mass is SPD, <x,y>:=x'*mass*y. 
%Output: Frobenius norm.

if nargin==1
    val=gather(real(sqrt(trace(R'*R))));
else
    val=gather(real(sqrt(trace(R'*mass*R))));
end

end