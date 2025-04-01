function [Rcur] = Rule2(train_data)

[data_num attribute_num] = size(train_data);
data=train_data(:,2:attribute_num);

nhan1=train_data(:,1);

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
    V_var = 0;
    min_value = min(train_data(:,feature_index));
    max_value = max(train_data(:,feature_index));

    delta = max_value - min_value;

    if cluster(feature_index) == 2
        V(1,1) = min_value - 0.5;
        V(2,1) = max_value;

    elseif cluster(feature_index) == 3
        V(1,1) = min_value;
        V(2,1) = min_value + delta/2;
        V(3,1) = max_value;

    else
        V(1,1) = min_value;
        V(2,1) = min_value + delta/4;
        V(3,1) = min_value + 2*delta/4;
        V(4,1) = min_value + 3*delta/4;
        V(5,1) = max_value;

    end
    [center,U] = FCM_Function(feature_data,cluster(feature_index),V,m,esp,maxTest);

    U = U';

    center_vector{feature_index}(:,1) = center(:,1);

    for i=1:data_num
        maximum = max(U(i,:));
        for j=1:cluster(feature_index)
            if (maximum == U(i,j))
                rules(i,feature_index) = j;
            end
            
        end
    end
    if feature_index ~= 1          
        center = center';
        centers{feature_index-1} = center(1,:);
        
    end
end

[t,sigma_M] = RuleWeight(rules, train_data,cluster,center_vector);

sigma_M(1,:) = [];                  
for i=1:(attribute_num-1)
    sigma_M(i,2:5) = sigma_M(i,1);
end

beta = zeros(data_num,attribute_num);
for i=1:data_num
    beta(i,:) = [[1 train_data(i,2:attribute_num)]\train_data(i,1)]';
end
label = train_data(:,1);
for i=1:data_num
    rules(i,(attribute_num+1)) = min(t(i,2:attribute_num));
end

for i=1:data_num-1
    for j=i+1:data_num
        if(rules(i,2:attribute_num) == rules(j,2:attribute_num))
            if(rules(i,(attribute_num+1)) > rules(j,(attribute_num+1)))
                rules(j,:)=0;
            else
                rules(i,:)=1;
            end
        end
    end
end
       

% Rules with weight < 0.5 will be removed
for i=1:data_num
    if(rules(i,(attribute_num+1)) < 0.5)
        rules(i,:) = 0;
    end
end
rules(:,1) = [];

% Filter rules
RuleCheck = zeros(1,attribute_num);
j = 1;
for i=1:data_num
    if (rules(i,:) ~= RuleCheck(1,:))
        FilteredRules(j,:) = [label(i) rules(i,1:(attribute_num-1))];
        
        FilterT(j,:)=t(i,:);
       
        j = j + 1;
    end
end
W_rule=min(FilterT');

ruleList = FilteredRules;

X1=ruleList;
% X1=[X1 nhan1];

label = train_data(:,1)+1;
label=label;
ruleList1=ruleList;
% ruleList1=rules;
%Tính nguy
[r1 r2]= size(ruleList1);

%Tinh trong so w
dem(3)=0
for i =2:r2
    dem(1)=0;dem(2)=0;dem(3)=0;
    for j=1:r1
        for l=1:cluster(i)
            if ruleList1(j,i)==l
                dem(l)=dem(l)+1;
            end
        end
    end
   
    w(i,:)=dem/r1;
end

% ruleList1
dem2=0;dem4=0;
wl(2)=0;
for i=1:r1
    if ruleList1(i,1)==2
        dem2=dem2+1;
    else
        dem4=dem4+1;
    end
end
wl(1)=dem2/r1;
wl(2)=dem4/r1;
m(r1,r2-1)=0;
for i=2: r2-1
    for j=1:r1
        m(j,i)=w(i,ruleList1(j,i))*w(i+1,ruleList1(j,i+1));
    end
end
for j=1:r1
    if ruleList1(j,1)==2
        m(j,1)=wl(1)*w(r2,ruleList1(j,r2));
    else
        m(j,1)=wl(2)*w(r2,ruleList1(j,r2));
    end
end
m1=sum(m')
tbc=mean(m1)
dem=1;
for i=1:r1
    if m1(i)>tbc
        Rcur(dem,:)=ruleList1(i,:);
        dem=dem+1;
    end
end
Rcur;
size(Rcur);
        
    
