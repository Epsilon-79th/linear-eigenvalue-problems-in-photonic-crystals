function flag=d_flag_bcc_single_gyroid(e)

g=@(r) sin(2*pi*r(1))*cos(2*pi*r(2))+sin(2*pi*r(2))*cos(2*pi*r(3))+sin(2*pi*r(3))*cos(2*pi*r(1));

if g(e)>1.1
    flag=1;
else
    flag=0;
end

end