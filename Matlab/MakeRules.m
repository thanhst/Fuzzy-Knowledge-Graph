function [centers centers_var t t_var] = MakeRules(train_dataset, var_data)

addpath('..\input');
train_data = load(['..\input\' train_dataset]);
var_data = [train_data(:,1) var_data];
[data_num attribute_num] = size(train_data);
cluster = zeros(1, attribute_num);                
cluster(1) = 2;                                     
for i=2:attribute_num
    temp = length(unique(train_data(:,i)));
    if temp == 2
        cluster(i) = 2;
    elseif temp == 3
        cluster(i) = 3;
    else
        cluster(i) = 5;
    end
end

m = 2;
esp = 0.01;
maxTest = 200;
center_vector = cell (1, attribute_num);
centers = cell(attribute_num-1, 1);
centers_var = cell(attribute_num-1, 1);

for feature_index=1:attribute_num
    feature_data = train_data(:,feature_index);
    V = 0;
    V_var = 0;
    min_value = min(train_data(:,feature_index));
    max_value = max(train_data(:,feature_index));
    min_value_var = min(var_data(:,feature_index));
    max_value_var = max(var_data(:,feature_index));
    delta = max_value - min_value;
    delta_var = max_value_var - min_value_var;
    if cluster(feature_index) == 2
        V(1,1) = min_value - 0.5;
        V(2,1) = max_value;
        V_var(1,1) = min_value_var - 0.5;
        V_var(2,1) = max_value_var;
    elseif cluster(feature_index) == 3
        V(1,1) = min_value;
        V(2,1) = min_value + delta/2;
        V(3,1) = max_value;
        V_var(1,1) = min_value_var;
        V_var(2,1) = min_value_var + delta_var/2;
        V_var(3,1) = max_value_var;
    else
        V(1,1) = min_value;
        V(2,1) = min_value + delta/4;
        V(3,1) = min_value + 2*delta/4;
        V(4,1) = min_value + 3*delta/4;
        V(5,1) = max_value;
        V_var(1,1) = min_value_var;
        V_var(2,1) = min_value_var + delta_var/4;
        V_var(3,1) = min_value_var + 2*delta_var/4;
        V_var(4,1) = min_value_var + 3*delta_var/4;
        V_var(5,1) = max_value_var;
    end
    [center,U] = FCM_Function(feature_data,cluster(feature_index),V,m,esp,maxTest);
    [center_var,U_var] = FCM_Function(feature_data,cluster(feature_index),V_var,m,esp,maxTest);
    U = U';
    U_var = U_var';
    center_vector{feature_index}(:,1) = center(:,1);
    center_vector_var{feature_index}(:,1) = center_var(:,1);
    for i=1:data_num
        maximum = max(U(i,:));
        maximum_var = max(U_var(i,:));
        for j=1:cluster(feature_index)
            if (maximum == U(i,j))
                rules(i,feature_index) = j;
            end
            if (maximum_var == U_var(i,j))
                rules_var(i,feature_index) = j;
            end
        end
    end
    if feature_index ~= 1          
        center = center';
        center_var = center_var';
        centers{feature_index-1} = center(1,:);
        centers_var{feature_index-1} = center_var(1,:);
    end
end

[t,sigma_M] = RuleWeight(rules, train_data,cluster,center_vector);
[t_var,sigma_M_var] = RuleWeight(rules_var, var_data,cluster,center_vector_var);
sigma_M(1,:) = [];                  
for i=1:(attribute_num-1)
    sigma_M(i,2:5) = sigma_M(i,1);
end
sigma_M_var(1,:) = [];                  
for i=1:(attribute_num-1)
    sigma_M_var(i,2:5) = sigma_M_var(i,1);
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
                rules(i,:)=0;
            end
        end
    end
end
       

Rules with weight < 0.5 will be removed
for i=1:data_num
    if(rules(i,(attribute_num+1)) < 0.5)
        rules(i,:) = 0;
    end
end
rules(:,1) = [];
rules_var(:,1) = [];
Filter rules
RuleCheck = zeros(1,attribute_num);
j = 1;
for i=1:data_num
    if (rules(i,:) ~= RuleCheck(1,:))
        FilteredRules(j,:) = [rules(i,1:(attribute_num-1)) label(i)];
        FilteredRules_var(j,:) = rules_var(i,:);
        FilterT(j,:)=t(i,:);
        FilterT_var(j,:)=t_var(i,:);
        j = j + 1;
    end
end
W_rule=min(FilterT');
W_rule_var=min(FilterT_var');
ruleList = FilteredRules;
ruleList_var = FilteredRules_var;
size(ruleList)
t_rule=min(t');
t_rule_var=min(t_var');
while A1>A
for i=1:size(rules)-1
     for j=i+1:size(rules)
         if rules(i,10)==rules(j,10)
             D(i,j)=calculate_D_Similarity(t_rule(i), t_rule_var(i), t_rule(j), t_rule_var(j));
         else
             D(i,j)=0;
         end
     end
end
D(size(rules),:)=0;
for i=1:size(rules)-1
     for j=i+1:size(rules)
         if rules(i,10)==rules(j,10)
            J(i,j)=calculate_J_Similarity(t_rule(i), t_rule_var(i), t_rule(j), t_rule_var(j));
         else
             J(i,j)=0;
         end
     end
end
J(size(rules),:)=0;
for i=1:size(rules)-1
     for j=i+1:size(rules)
         if rules(i,10)==rules(j,10)
            C(i,j)=calculate_C_Similarity(t_rule(i), t_rule_var(i), t_rule(j), t_rule_var(j));
         else
             C(i,j)=0;
         end
     end
end
C(size(rules),:)=0;
nhan2=0;
gtri2=0;
nhan4=0;
gtri4=0;
for i=1:size(rules)-1
     for j=i+1:size(rules)
         if rules(i,10)==2
            F(i,j)=0.2*D(i,j)+0.5*J(i,j)+0.3*C(i,j);
            nhan2=nhan2+1;
            gtri2=gtri2+F(i,j);
         else
            F(i,j)=0.5*D(i,j)+0.3*J(i,j)+0.2*C(i,j);
            nhan4=nhan4+1;
            gtri4=gtri4+F(i,j);
         end
     end
end
F(size(rules),:)=0;
tb2=gtri2/nhan2;
tb4=gtri4/nhan4;
for i=1:size(rules)-1
     for j=i+1:size(rules)
         if ((rules(i,10)==2) && (F(i,j)<tb2)) || ((rules(i,10)~=2) && (F(i,j)<tb4))
             F(i,j)=0;
         end
     end
end
    for i=1:size(rules)-1
         for j=i+1:size(rules)
             if (F(i,j)~=0)
                F(j,:)=0;
             end
        end
    end
    s=1;
    for i=2:size(rules)
        if F(i,:)==0
            continue;
        else
            FilteredRules(s,:) = [rules(i,1:(attribute_num-1)) label(i)];
            FilteredRules_var(s,:) = rules_var(i,:);
            FilterT(s,:)=t(i,:);
            FilterT_var(s,:)=t_var(i,:);
            s=s+1;
        end
    end
end
W_rule=min(FilterT');
W_rule_var=min(FilterT_var');
ruleList = FilteredRules;
ruleList_var = FilteredRules_var;
filename = strrep(train_dataset,'.txt','.mat');
filename = strrep(filename, 'Database', 'FIS');
addpath('..\output');
save(['..\output\RuleList.mat'], 'ruleList');
save(['..\output\RuleList.mat'], 'ruleList_var','-append');
save(['..\output\' filename], 'sigma_M');
save(['..\output\' filename], 'sigma_M_var', '-append');
fprintf('==============================================================================\n');
fprintf('Rule Generation process is done. RuleList.mat created. \n');
% fprintf('Running CROSS_VALIDATION.m... \n');
fprintf('==============================================================================\n');