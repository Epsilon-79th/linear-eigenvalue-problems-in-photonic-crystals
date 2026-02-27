function flag=d_flag_fcc(e)

a1=[0;1/2;1/2];
a2=[1/2;0;1/2];
a3=[1/2;1/2;0];

cnt=(a1+a2+a3)/4;

tran1=[zeros(3,1),eye(3),[0;1;1],[1;0;1],[1;1;0],[1;1;1],...
       [0;1/2;1/2],[1/2;0;1/2],[1/2;1/2;0],[1;1/2;1/2],[1/2;1;1/2],[1/2;1/2;1],...
       cnt,cnt+a1,cnt+a2,cnt+a3];
tran2=[zeros(3,1),a1,a2,a3]; 

r=0.12;
b=0.11;

o1=cnt/2;           d1=cnt/2;        c1=norm(d1); d1=d1/c1;
o2=(a1+cnt)/2;      d2=(a1-cnt)/2;   c2=norm(d2); d2=d2/c2;
o3=(a2+cnt)/2;      d3=(a2-cnt)/2;   c3=norm(d3); d3=d3/c3;
o4=(a3+cnt)/2;      d4=(a3-cnt)/2;   c4=norm(d4); d4=d4/c4;

X=e-tran1;

if (diag(X'*X)-r^2)<0 || ...
   ell(e,o1,b,c1,d1,tran2) || ...
   ell(e,o2,b,c2,d2,tran2) || ...
   ell(e,o3,b,c3,d3,tran2) || ...
   ell(e,o4,b,c4,d4,tran2)
    flag=1;
else
    flag=0;
end

end

function val=ell(x,cnt,b,c,d,tran)

X=x-(cnt+tran);
a=sqrt(b^2+c^2);
L1=(X'*d).^2;
L2=diag(X'*X)-L1;

if min(L1/(a^2)+L2/(b^2))<1
   val=1;
else
    val=0;
end

end