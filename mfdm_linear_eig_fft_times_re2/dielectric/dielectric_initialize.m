function dielectric_initialize(D_name)

% Initialization: crystal information preloaded.

CT_sc=eye(3);
CT_bcc=[0 1 1;1 0 1;1 1 0];
CT_fcc=[-1,1,1;1,-1,1;1,1,-1];
sym_sc=[[0;0;0],[pi;0;0],[pi;pi;0],[pi;pi;pi],[0;0;0]];
sym_bcc=[[0;0;2*pi],[0;0;0],[pi;pi;pi],[0;0;2*pi],[pi;0;pi],...
         [0;0;0],[0;2*pi;0],[pi;pi;pi],[pi;0;pi]];
sym_fcc=[[0;2*pi;0],[pi/2;2*pi;pi/2],[pi;pi;pi],[0;0;0],...
         [0;2*pi;0],[pi;2*pi;0],[3*pi/2;3*pi/2;0]];

if nargin==1
    save([D_name,'/diel_info.mat'],'CT_sc','CT_bcc','CT_fcc',...
     'sym_sc','sym_bcc','sym_fcc');
else
    save('diel_info.mat','CT_sc','CT_bcc','CT_fcc',...
     'sym_sc','sym_bcc','sym_fcc');
end

end