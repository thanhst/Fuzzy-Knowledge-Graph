% This script is to generate a set of rules for FIS.
% 
% Input:
% 1. full_data: all dental records for determining minimum and maximum
% value of each feature
% 2. train_data: training dataset for rules generation, train_data can be
% the same with full_data 
%
% Output:
% 1. ruleList: a set of rule
% 2. sigma_M: array of 5 Sigmas for later use  corresponding 5 featueres
% (Sigma of Label is not in this output)
% 3. centers: array of 5 centers corresponding 5 featueres for later use
% (Center of Label is not in this output)
%
% 3 outputs will be saved to FIS_para.mat

clear;
full_data = 'dental.txt';
train_data = 'dental.txt';

full_data=load(full_data);
minEEI=min(full_data(:,2));
maxEEI=max(full_data(:,2));
minLBP=min(full_data(:,3));
maxLBP=max(full_data(:,3));
minRGB=min(full_data(:,4));
maxRGB=max(full_data(:,4));
minGradient=min(full_data(:,5));
maxGradient=max(full_data(:,5));
minPatch=min(full_data(:,6));
maxPatch=max(full_data(:,6));
minLabel=min(full_data(:,7));
maxLabel=max(full_data(:,7));

train_data=load(train_data);
h=size(train_data,1);

cluster=[3 3 3 3 3 5];
m=2;
esp=0.01;
maxTest=200;

%Feature 1 Entropy - Edge - Intensity
feature1=train_data(:,2);
V=0;   
V(1,1)=minEEI;
V(2,1)=(minEEI+maxEEI)/2;
V(3,1)=maxEEI;
[center,U] = FCM_Function(feature1,cluster(1,1),V,m,esp,maxTest);
U=U';
center1(:,1)=center(:,1);
for i=1:h
    maximum=max(U(i,:));
    for j=1:cluster(1,1)
        if(maximum==U(i,j)) rules(i,1)=j;
        end
    end
end
center=center';
centers(1,:)=center(1,:);

%Feature 2 LBP
feature2=train_data(:,3);
V=0;    
V(1,:)=minLBP;
V(2,:)=(maxLBP+minLBP)/2;
V(3,:)=maxLBP;
[center,U] = FCM_Function(feature2,cluster(1,2),V,m,esp,maxTest);
U=U';
center2(:,1)=center(:,1);
for i=1:h
    maximum=max(U(i,:));
    for j=1:cluster(1,2)
        if(maximum==U(i,j)) rules(i,2)=j;
        end
    end
end
center=center';
centers(2,:)=center(1,:);

%Feature 3 RGB
feature3=train_data(:,4);
V=0;    
V(1,:)=minRGB;
V(2,:)=(maxRGB+minRGB)/2;
V(3,:)=maxRGB;
[center,U] = FCM_Function(feature3,cluster(1,3),V,m,esp,maxTest);
U=U';
center3(:,1)=center(:,1);
for i=1:h
    maximum=max(U(i,:));
    for j=1:cluster(1,3)
        if(maximum==U(i,j)) rules(i,3)=j;
        end
    end
end
center=center';
centers(3,:)=center(1,:);

%Feature 4 Gradient
feature4=train_data(:,5);
V=0;    
V(1,:)=minGradient;
V(2,:)=(minGradient+maxGradient)/2;
V(3,:)=maxGradient;
[center,U] = FCM_Function(feature4,cluster(1,4),V,m,esp,maxTest);
U=U';
center4(:,1)=center(:,1);
for i=1:h
    maximum=max(U(i,:));
    for j=1:cluster(1,4)
        if(maximum==U(i,j)) rules(i,4)=j;
        end
    end
end
center=center';
centers(4,:)=center(1,:);

%Feature 5 Patch
feature5=train_data(:,6);
V=0;
V(1,:)=minPatch;
V(2,:)=(minPatch+maxPatch)/2;
V(3,:)=maxPatch;   
[center,U] = FCM_Function(feature5,cluster(1,5),V,m,esp,maxTest);
U=U';
center5(:,1)=center(:,1);
for i=1:h
    maximum=max(U(i,:));
    for j=1:cluster(1,5)
        if(maximum==U(i,j)) rules(i,5)=j;
        end
    end
end
center=center';
centers(5,:)=center(1,:);

%Label - Dental Problem
label=train_data(:,7);
V=0;
seg=(maxLabel-minLabel)/4;
V(1,:)=minLabel;
V(2,:)=V(1,:)+seg;
V(3,:)=V(2,:)+seg;
V(4,:)=V(3,:)+seg; 
V(5,:)=maxLabel; 
[center,U] = FCM_Function(label,cluster(1,6),V,m,esp,maxTest);
U=U';
center6(:,1)=center(:,1);
for i=1:h
    maximum=max(U(i,:));
    for j=1:cluster(1,6)
        if(maximum==U(i,j)) rules(i,6)=j;
        end
    end
end

%Get weight of membership
[t,sigma_M] = RuleWeight(rules, train_data,cluster,center1, center2,...
                 center3, center4, center5, center6);

%Create sigma_M for FIS
sigma_M(6,:)=[];
for i=1:5
    sigma_M(i,2:3)=sigma_M(i,1);
end

%Get weight for each rule
for i=1:h
    rules(i,7)=min(t(i,:));
    rules(i,8)=train_data(i,7);
end

%Remove weaker or duplicate rules
for i=1:h-1
    for j=i+1:h
        if(rules(i,1:5)==rules(j,1:5))
            if(rules(i,7)>rules(j,7))
                rules(j,:)=0;
            else
                rules(i,:)=0;
            end
        end
    end
end

%Rules with weight < 0.5 will be removed
for i=1:h
    if(rules(i,7)<0.5)
        rules(i,:)=0;
    end
end

%Filter rules
RuleCheck=[0 0 0 0 0 0 0 0];
j=1;
for i=1:h        
    if (rules(i,:)~=RuleCheck(1,:))
        FilteredRules(j,:)=rules(i,:);
        j=j+1;
    end
end
ruleList=FilteredRules(:,1:6);

save RuleList ruleList;
save 'FIS_para.mat' sigma_M;
save 'FIS_para.mat' centers -append;
% clear;clc;
% fprintf('==============================================================================\n');
% fprintf('  Rule Generation process is done. RuleList.mat created. Run MakeData.m next. \n');
% fprintf('==============================================================================\n');