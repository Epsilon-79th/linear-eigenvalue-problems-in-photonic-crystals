function AX=A_fft(X,D)

% Matrix multiplication done by FFT.
% A=[[0,-D3,D2];[D3,0,-D1],[-D2,D1,0]], return AX=A*X.

[n,m]=size(X);
n=round(n/3);
N=round(n^(1/3));

AX=reshape(X,[N N N 3*m]);
for d=1:3
    AX=fft(AX,[],d);
end

AX=reshape(AX,[3*n,m]);

%{
AX1=-D(2*n+1:end).*AX(n+1:2*n,:)+D(n+1:2*n).*AX(2*n+1:end,:);
AX2=D(2*n+1:end).*AX(1:n,:)-D(1:n).*AX(2*n+1:end,:);
AX3=-D(n+1:2*n).*AX(1:n,:)+D(1:n).*AX(n+1:2*n,:);

AX=reshape([AX1;AX2;AX3],[N N N 3*m]);
%}

AX=reshape([-D(2*n+1:end).*AX(n+1:2*n,:)+D(n+1:2*n).*AX(2*n+1:end,:); ...
            D(2*n+1:end).*AX(1:n,:)-D(1:n).*AX(2*n+1:end,:); ...
            -D(n+1:2*n).*AX(1:n,:)+D(1:n).*AX(n+1:2*n,:)],[N N N 3*m]);

for d=1:3
    AX=ifft(AX,[],d);
end

AX=reshape(AX,[3*n,m]);

end