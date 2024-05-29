function flag=d_flag_sc_curv(e)

R1=0.345; r1=0.11;
e=e-0.5;

if norm(e)<=R1 ||...
   norm(e([2 3]))<=r1 ||...
   norm(e([1 3]))<=r1 ||...
   norm(e([1 2]))<=r1
    flag=1;
else
    flag=0;
end

end