function flag=d_flag_sc_flat_1(e)

e=mod(e,1);
if (e(2)<=0.25 && e(3)<=0.25) ||...
   (e(1)<=0.25 && e(3)<=0.25) ||...
   (e(1)<=0.25 && e(2)<=0.25)  
    flag=1;
else
    flag=0;
end

end