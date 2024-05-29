function HX=H_fft(X,D)

% Matrix multiplication done by FFT.
% H=FDF^{-1}, F: discrete Fourier transform.

[n,m]=size(X);
n=round(n/3);
N=round(n^(1/3));

HX=reshape(X,[N N N 3*m]);
for d=1:3
    HX=fft(HX,[],d);
end

HX=reshape(D*reshape(HX,[3*n,m]),[N N N 3*m]);
for d=1:3
    HX=ifft(HX,[],d);
end

HX=reshape(HX,[3*n,m]);

end