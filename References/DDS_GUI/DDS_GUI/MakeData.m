clear;
data=load('dental.txt');
order=data(:,1);
data(:,1)=[];

train_input=load('training34a.txt');
train_input(:,1)=[];
train_output=train_input(:,6);
train_input(:,6)=[];

load resultFeature.mat;
%num=35;

test_input=resultFeature;
%test_input(:,1)=[];
%test_output=test_input(:,6);
%test_input(:,6)=[];

save data data;
save data train_input -append;
save data train_output -append;
save data test_input -append;
%save data test_output -append;

% clc;
% fprintf('============================================\n');
% fprintf('  Data.mat created. Run DENTAL_FIS.m next.  \n');
% fprintf('============================================\n');