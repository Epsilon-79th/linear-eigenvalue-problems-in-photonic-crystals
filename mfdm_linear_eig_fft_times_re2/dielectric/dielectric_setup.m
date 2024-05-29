function [M1,sym_points,CT]=dielectric_setup(N,d_filename)

%% Input:
% N: grid size.
% eps: dielectric coefficient.
% d_filename: lattice type, d_flag name.

%% Output:
% M1: indexes.
% sym_points: symmetry points in the 1st Brillouin zone.
% CT: matrix of coordinate change.

%%

tic;

D_name='dielectric';
dir_name=[D_name,'/dielectric_examples'];
info_name=[D_name,'/diel_info'];

if ~exist(dir_name,'dir')
    mkdir(dir_name);
end

if ~exist(info_name,'file')
    dielectric_initialize(D_name);
end

if d_filename(1)=='s'
    d_flag_type='sc';
else
    d_flag_type=d_filename(1:3);
end

CT=getfield(load(info_name,['CT_',d_flag_type]),['CT_',d_flag_type]);
sym_points=getfield(load('diel_info.mat',['sym_',d_flag_type]),['sym_',d_flag_type]);

if exist([D_name,'/d_flag_',d_filename],'file')
    eval(['d_flag_option=@d_flag_',d_filename,';']);
else
    error("Input distance function type doesn't exist.")
end

ind_name=[d_filename,'_',num2str(N)];
d_filename=[dir_name,'/',d_filename];

if exist([d_filename,'.mat'],'file')
    % Judge whether '...' contains certain variable.
    var_names=who('-file',d_filename);

    flag_exist=false;
    for i=1:length(var_names)
        flag_exist=strcmp(var_names{i},ind_name);
        if flag_exist
            break;
        end
    end

    if ~flag_exist
        fprintf('New grid size=%d for "%s" lattice.\n',N,d_filename);
        M1=dielectric_index(N,inv(CT),d_flag_option);
        eval([ind_name,'=M1;']);
        save(d_filename,ind_name,'-append');
    else
        fprintf('Lattice type "%s" with grid size=%d already exists.\n',d_filename,N);
        M1=getfield(load(d_filename,ind_name),ind_name);
    end
else
    fprintf('New lattice type "%s".\n',d_filename);
    M1=dielectric_index(N,inv(CT),d_flag_option);
    eval([ind_name,'=M1;']);
    save(d_filename,ind_name);
end

t=toc;
fprintf('time for setting up the dielectric coefficient=%gs.\n',t);

end