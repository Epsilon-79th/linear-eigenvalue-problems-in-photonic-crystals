function plot_bandgap(N,lattice_type)

%% Loading ...

file_name=['output/bandgap_',lattice_type,'.mat'];

% .mat file doesn't exist.
if ~exist(file_name,'file')
    error("Can't find file '%s'.\n",file_name);
end

var_names=who('-file',file_name);
ind_name=[lattice_type,'_',num2str(N)];
flag_exist=false;
for i=1:length(var_names)
    flag_exist=strcmp(var_names{i},ind_name);
    if flag_exist
        break;
    end
end

% Grid size hasn't been computed.
if ~flag_exist
    error("Grid size N=%d hasn't been computed for lattice type '%s'.\n",N,lattice_type);
end

gap_info=getfield(load(file_name,ind_name),ind_name);

%% Setting the symmetry points.
if lattice_type(1)=='s'
    d_flag_type='sc';
else
    d_flag_type=lattice_type(1:3);
end
sym_points=getfield(load('dielectric/diel_info.mat',['sym_',d_flag_type]),['sym_',d_flag_type]);

[n,m]=size(gap_info.eigen);

[~,n_pt]=size(sym_points); n_pt=n_pt-1;
gap=round(n/n_pt);
Alpha=zeros(3,n);
for i=1:n_pt
    Alpha(:,i*gap)=sym_points(:,i+1);
    for j=1:gap-1
        Alpha(:,(i-1)*gap+j)=(j*sym_points(:,i+1)+(gap-j)*sym_points(:,i))/gap;
    end
end

%% Plotting ...

for i=1:n 
    x=i*ones(1,m);
    y=real(gap_info.eigen(i,:));
    if norm(y)>0
        scatter(x,y,'b.');
        hold on;
    end
end

hold off;
xticks([0,10,20,30,40]*2);

xticklabels({'K','L','M','N','K'});
ylim([0,0.8]);
ylabel('\omega/2\pi');

title(['Figure of bandgap, lattice type: ',lattice_type,', grid size N=',num2str(N),'.']);

end