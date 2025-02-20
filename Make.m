function [X1 A B cluster center_vector] = Make(train_data)

[data_num attribute_num] = size(train_data);
data=train_data(:,1:attribute_num-1);
nhan1=train_data(:,attribute_num);

cluster = zeros(1, attribute_num);                  
                                     
for i=1:attribute_num
    temp = length(unique(train_data(:,i)));
    if temp == 2
        cluster(i) = 2;
    elseif temp == 3
        cluster(i) = 3;
    else
        cluster(i) = 3;
    end
end

m = 2;
esp = 0.01;
maxTest = 200;
center_vector = cell (1, attribute_num);
centers = cell(attribute_num-1, 1);
for feature_index=1:attribute_num
    feature_data = train_data(:,feature_index);
    V = 0;
    min_value = min(train_data(:,feature_index));
    max_value = max(train_data(:,feature_index));
    delta = max_value - min_value;
    if cluster(feature_index) == 2
        V(1,1) = min_value - 0.5;
        V(2,1) = max_value;

%     elseif cluster(feature_index) == 3
    else
        V(1,1) = min_value;
        V(2,1) = min_value + delta/2;
        V(3,1) = max_value;
  
    end
    [center,U] = FCM_Function(feature_data,cluster(feature_index),V,m,esp,maxTest);
    U = U';
    center_vector{feature_index}(:,1) = center(:,1);
    for i=1:data_num
        maximum = max(U(i,:));
        for j=1:cluster(feature_index)
            if (maximum == U(i,j))
                X1(i,feature_index) = j;
            end
        end
    end
    
end
% X1=[X1 nhan1];

label = train_data(:,1)+1;
label=label;
ruleList1=X1;
% T1=0;
% T2=0;
% for i=1:data_num
%     if ruleList1(i,attribute_num)==1
%         if label(i)==1
%             T1=T1+1;
%         else
%             T2=T2+1;
%         end
%     end
% end
% ruleList1(:,attribute_num);
% if T1<T2
%     for i=1:data_num
%         if ruleList1(i,attribute_num)==1
%             ruleList1(i,attribute_num)=2;
%         else
%             ruleList1(i,attribute_num)=1;
%         end
%     end
% end



% % -------------------
% 
ruleList1=[1 1 1 1 1 1;2 3 2 2 2 2;1 3 1 1 1 1;1 1 1 2 1 1;2 3 2 3 2 2;2 3 3 3 3 2];
% 
[r1 r2]= size(ruleList1);
%Sinh do thi 
%Tinh A
for i =1:r1
    j=1;
    for l=1:r2-3
        for h=l+1:r2-2
            for s=h+1:r2-1
                dem=0;
                for t=1:r1
                    if (ruleList1(t,l)==ruleList1(i,l))&(ruleList1(t,h)==ruleList1(i,h))&(ruleList1(t,s)==ruleList1(i,s))
                            dem=dem+1;
                    end
                end
                
            A(i,j)=dem/r1;
            j=j+1;
            end
        end
    end
end
ruleList1;
A
% C=sum(A');
%Tinh tong A theo nhãn lu?t
[k1 k2]= size(A);
T=sum(A')
% for i=1:2
%     for j=1:k2
%         tong=0;
%         for l=1:k1
%             if ruleList1(l,r2)==i
%                 tong=tong+A(l,j);
%             end
%         end
%         T(i,j)=tong;
%     end
% end
% T
%tìm B
r1;
for i=1: r1
    j=1;
    for l=1:r2-2
        for h=l+1:r2-1
%             for s=h+1:r2-1
                dem1=0;
                dem2=0;
%                 dem3=0;
                for t=1:r1
                    if (ruleList1(t,l)==ruleList1(i,l))&(ruleList1(t,r2)==ruleList1(i,r2))
                        dem1=dem1+1;
                    end
                    if (ruleList1(t,h)==ruleList1(i,h))&(ruleList1(t,r2)==ruleList1(i,r2))
                        dem2=dem2+1;
                    end
%                     if (ruleList1(t,s)==ruleList1(i,s))&(ruleList1(t,r2)==ruleList1(i,r2))
%                         dem3=dem3+1;
%                     end
                end
                M1=dem1/r1;
                M2=dem2/r1;
%                 M3=dem3/r1;
              
%                 for l1=1:2
%                     if ruleList1(i,r2)==l1
                B(i,j)=T(i)*min(M1,M2);
                j=j+1;
%             end
        end
    end
end
B
   

% nhan=[1 2];
% n1=size(nhan)
% for i=1:r1
%     for j=1:r2
%         dem=0;
%         for l=1:r1
%             if (ruleList1(i,r2)==ruleList1(l,r2))&(ruleList1(i,j)==ruleList1(l,j))
%                 dem=dem+1;
%             end
%         end
%         
%         B(i,j)=C(i)*dem/r1;
%     end
% end
size(A);
size(B);
% 
% 
% 
% 
% filename = strrep(train_data,'.txt','.mat');
% filename = strrep(filename, 'Database', 'FIS');
% addpath('..\output');
% save(['..\output\RuleList.mat'], 'ruleList');
% save(['..\output\RuleList.mat'], 'ruleList_var','-append');
% save(['..\output\FIS_Breast.mat'], 'sigma_M');
% save(['..\output\FIS_Breast.mat'], 'sigma_M_var', '-append');

