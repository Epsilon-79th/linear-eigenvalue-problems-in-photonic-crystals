function D0=find_eig(sten1,sten2,N)

%return the eigenvalues of a circulant matrix.

D0=zeros(N,1);
n1=length(sten1);
n2=length(sten2);
for i=1:N
    for j=1:n1
        D0(i)=D0(i)+sten1(j)*exp(2*pi*sqrt(-1)*(i-1)*(j-1)/N);
    end
    
    for j=1:n2
        D0(i)=D0(i)+sten2(j)*exp(2*pi*sqrt(-1)*(i-1)*(N-j)/N);
    end
end

end
