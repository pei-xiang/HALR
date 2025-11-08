clear;
close all;
clc;

load abu-urban-2.mat;
img_src=data;
mask=map;
sz = size(img_src);
img_src = double(img_src);
img_src=reshape(img_src,sz(1)*sz(2),sz(3));
img_src=img_src';
MM=sz(1)*sz(2);

% parameter
r = 10;   % try r=2, 4, or 8, 10
eps = 0.3^2; % try eps=0.1^2, 0.2^2, 0.4^2
lambda = 0.00001;       % tradeoff parameter

%% harmonic analysis

A0_2=mean(img_src,1);
A0_2=reshape(A0_2,sz(1),sz(2));     %% harmonic remainder
A0_2=mat2gray(A0_2);
figure;
imshow(A0_2);axis image;title('{A_0}/2');

% Ah Bh Ch faih
img_src=reshape(img_src',sz(1),sz(2),sz(3));

datasize=sz;
ha_num=5;                          %% number of harmonic analysis
output_Ah=zeros(datasize(1),datasize(2),2*ha_num-1);
output_Bh=zeros(datasize(1),datasize(2),2*ha_num-1);
output_Ch=zeros(datasize(1),datasize(2),2*ha_num-1);
output_faih=zeros(datasize(1),datasize(2),2*ha_num-1);
for ha_times=1:ha_num
    for j=1:datasize(2)
        for i=1:datasize(1)
            for k=1:ha_times
                output_Ah(i,j,k)=function_Ah(img_src(i,j,:),k,datasize(3));
                output_Bh(i,j,k)=function_Bh(img_src(i,j,:),k,datasize(3));
                output_Ch(i,j,k)=sqrt(output_Ah(i,j,k)^2+output_Bh(i,j,k)^2);   %%  amplitude
                output_faih(i,j,k)=atan(output_Ah(i,j,k)/output_Bh(i,j,k));     %%  phase
            end
            
        end
    end
    HA=zeros(size(output_Ah,1),size(output_Ah,2),2*ha_times+1);
    for i=1:size(output_Ah,1)
        for j=1:size(output_Ah,2)
            temp1=output_Ch(i,j,1:ha_times);
            temp2=output_faih(i,j,1:ha_times);
            temp=[A0_2(i,j);temp1(:);temp2(:)];
            HA(i,j,:)=temp(:);      % reconstruct
        end
    end
    % output_name=sprintf('%s%d%s','HA_',2*ha_times+1,'.mat');
    % save(output_name,'HA');
    % 
    % % plot the 1th HA analysis
    % if ha_times==1
    %     figure;
    %     imshow(mat2gray(HA(:,:,2)));axis image;title('C_1');
    %     figure;
    %     imshow(mat2gray(HA(:,:,3)));axis image;title('\phi_1');
    % end
end

%% Guided Filtering
I=HA(:,:,1);
q=zeros(sz(1),sz(2),ha_num);
HAT=zeros(sz(1),sz(2),ha_num+1);
HAT(:,:,1)=I;
for ii=1:ha_num
    p=HA(:,:,ii+1);
    q(:,:,ii) = guidedfilter(I, p, r, eps);
    HAT(:,:,ii+1)=abs(q(:,:,ii)-p);
end
re1=zeros(sz(1),sz(2));
for ii=1:ha_num
    re1=re1+HAT(:,:,ii+1);
end
re2=re1./ha_num;

%% Dictionary Construction
HAF=reshape(HAT,sz(1)*sz(2),ha_num+1);
HAF=HAF';
sort_d=re2(:);
k=length(sort_d);
kk=k*0.95;                               % the top 90% of pixels after sorting
[sort_s,ix]=sort(sort_d,'ascend'); 
[row,col]=ind2sub(size(re2),ix);
dict = [];
sub=zeros(ha_num+1,1);
for ik=1:kk
    te=rand(1);
    if te<0.001                               % the proportion of atoms in the dictionary
        sub(:,1)=HAT(row(ik),col(ik),:);
        dict=[dict sub];                     % background dictionary
    end
end

% Dictionary based low-rank decomposition 
for k1=1:length(lambda)
    [A_hat,E_hat] = inexact_alm_lrr_l1(HAF,dict,lambda(k1),0,0);
end
nom=reshape(E_hat',sz(1),sz(2),ha_num+1);

%% anomaly extraction
result1=zeros(sz(1),sz(2));
for jj=1:sz(1)
    for v=1:sz(2)
        xy=nom(jj,v,:);
        xy=xy(:);
        result1(jj,v)=sqrt(xy'*xy);
    end
end

result2=result1;
figure;
imshow(mat2gray(result2),'colormap',parula);


%% ROC curve
disp('Running ROC...');
r4=reshape(result2,1,MM);
mask = reshape(mask, 1, MM);
anomaly_map = logical(double(mask)>=1);
normal_map = logical(double(mask)==0);
r_max = max(r4(:));
taus = linspace(0, r_max, 5000);
for index2 = 1:length(taus)
  tau = taus(index2);
  anomaly_map_rx = (r4 > tau);
  PF2(index2) = sum(anomaly_map_rx & normal_map)/sum(normal_map);
  PD2(index2) = sum(anomaly_map_rx & anomaly_map)/sum(anomaly_map);
end
area4 =  sum((PF2(1:end-1)-PF2(2:end)).*(PD2(2:end)+PD2(1:end-1))/2);
figure;
plot(PF2, PD2, 'k-', 'LineWidth', 2);  grid on
xlabel('False alarm rate'); ylabel('Probability of detection');
legend('HALR');
axis([0 1 0 1])