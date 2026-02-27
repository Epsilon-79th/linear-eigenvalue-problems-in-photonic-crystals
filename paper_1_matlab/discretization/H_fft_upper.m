function HX=H_fft_upper(X,D,SD)

% Matrix multiplication done by FFT.
% H=FDF^{-1}, F: discrete Fourier transform.

[n,m]=size(X);
n=round(n/3);
N=round(n^(1/3));

HX=reshape(X,[N N N 3*m]);
for d=1:3
    HX=fft(HX,[],d);
end

HX=reshape(HX,[3*n,m]);

if nargin==3
    HX=reshape([D(1:n).*HX(1:n,:)+SD(1:n).*HX(n+1:2*n,:)+SD(n+1:2*n).*HX(2*n+1:end,:);...
                conj(SD(1:n)).*HX(1:n,:)+D(n+1:2*n).*HX(n+1:2*n,:)+SD(2*n+1:end).*HX(2*n+1:end,:);...
                conj(SD(n+1:2*n)).*HX(1:n,:)+conj(SD(2*n+1:end)).*HX(n+1:2*n,:)+D(2*n+1:end).*HX(2*n+1:end,:)],[N,N,N,3*m]);
else
    HX=reshape([D(1:n).*HX(1:n,:)+D(3*n+1:4*n).*HX(n+1:2*n,:)+D(4*n+1:5*n).*HX(2*n+1:end,:);...
                conj(D(3*n+1:4*n)).*HX(1:n,:)+D(n+1:2*n).*HX(n+1:2*n,:)+D(5*n+1:end).*HX(2*n+1:end,:);...
                conj(D(4*n+1:5*n)).*HX(1:n,:)+conj(D(5*n+1:end)).*HX(n+1:2*n,:)+D(2*n+1:3*n).*HX(2*n+1:end,:)],[N N N 3*m]);
end

for d=1:3
    HX=ifft(HX,[],d);
end

HX=reshape(HX,[3*n,m]);

end