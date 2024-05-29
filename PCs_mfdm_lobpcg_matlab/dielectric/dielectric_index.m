function ind_d=dielectric_index(N,CT,d_flag)

%% Input:
% N: grid size. CT: transition matrix.  
% d_flag:  bool function handle.

%% Output: 
% ind_d: index related to material. 

%% Initialization.

n=N^3;
h=1/N;
ind_d=zeros(1,3*n);

ind=0;L=0;

%% Loop.

for z=1:N
    for y=1:N
        for x=1:N
            ind=ind+1;
            e_x=CT*[(x-1/2)*h;(y-1)*h;(z-1)*h];
            e_y=CT*[(x-1)*h;(y-1/2)*h;(z-1)*h];
            e_z=CT*[(x-1)*h;(y-1)*h;(z-1/2)*h];
            if d_flag(e_x)
                L=L+1;ind_d(L)=ind;
            end
            
            if d_flag(e_y)
                L=L+1;ind_d(L)=ind+n;
            end
            
            if d_flag(e_z)
                L=L+1;ind_d(L)=ind+2*n;
            end

        end        
    end
end

ind_d=ind_d(1:L);

end


