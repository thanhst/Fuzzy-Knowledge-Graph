function [Time, Accuracy, Recall, Precision] = Run_Train_FIS(temp_data, var_data, centers, centers_var,ruleList, ruleList_var)
% Complex weighted error
% clearing rate
eta=0.000001;
% output_order=1;
term_num=5;
FIS_para_filename = strrep('Train','.txt','.mat');
FIS_para_filename = strrep(FIS_para_filename, 'Database', 'FIS');
addpath('..\output');
load(['..\output\' FIS_para_filename]);
% load(['..\output\RuleList.mat']);

% addpath('..\input');
% temp_data = load(['..\input\' database]);
train_output = temp_data(:,1);
temp_data(:,1) = [];
train_input = temp_data;
input_num=size(train_input,1);
attri_num=size(train_input,2);
rule_num=size(ruleList,1);
rule_length=size(ruleList,2);
re_degree_M=ones(input_num,attri_num,term_num);
im_degree_M=ones(input_num,attri_num,term_num);
wtsum_re = zeros(input_num,1);
wtsum_im = zeros(input_num,1);
M_DataPerRule_re = zeros(input_num,rule_num);
M_DataPerRule_im = zeros(input_num,rule_num);

start_time_train = cputime;
for i=1:input_num
    for j=1:attri_num
        for k=1:length(centers{j})
            re_degree_M(i,j,k) = gaussmf(train_input(i,j),[sigma_M(j,k) centers{j}(k)]);
            im_degree_M(i,j,k) = abs(-(gaussmf(var_data(i,j),[sigma_M_var(j,k) centers_var{j}(k)]))*(var_data(i,j) - centers_var{j}(k))/(sigma_M_var(j,k)^2));
        end
    end
end
re_min_M=1;
im_min_M=1;
% size(re_degree_M)
% input_num
% rule_num
% ruleList
for i=1:input_num
    for j=1:rule_num
        re_min_M=1;
        im_min_M=min(im_degree_M(i,1,:));
        for k=1:attri_num
            re_degree_M(i,k,ruleList(j,k));
            if re_min_M>re_degree_M(i,k,ruleList(j,k));
                re_min_M=re_degree_M(i,k,ruleList(j,k));
            end
            if im_min_M>im_degree_M(i,k,ruleList_var(j,k));
                im_min_M=im_degree_M(i,k,ruleList_var(j,k));
            end
        end
        M_DataPerRule_re(i,j)=re_min_M;
        M_DataPerRule_im(i,j)=im_min_M;
    end 
end

result=zeros(input_num,1);
result_re=zeros(input_num,1);
result_im=zeros(input_num,1);
for i=1:input_num
    result_re(i)=0; result_im(i)=0; wtsum_re(i)=0; wtsum_im(i)=0;
    for j=1:rule_num
        wtsum_re(i)=wtsum_re(i)+M_DataPerRule_re(i,j)*cos(M_DataPerRule_im(i,j));
        wtsum_im(i)=wtsum_im(i)+M_DataPerRule_re(i,j)*sin(M_DataPerRule_im(i,j));
        result_re(i)=result_re(i)+M_DataPerRule_re(i,j)*cos(M_DataPerRule_im(i,j))*ruleList(j,rule_length);
        result_im(i)=result_im(i)+M_DataPerRule_re(i,j)*sin(M_DataPerRule_im(i,j))*ruleList(j,rule_length);
    end
    result(i)=sqrt((result_re(i)^2 + result_im(i)^2)/(wtsum_re(i)^2 + wtsum_im(i)^2));
end

% temp=round(result);
temp=result;
data_num=size(result);
for i=1:data_num
    if temp(i)>=0.41*(max(train_output)+min(train_output))
        temp(i)=max(train_output);
    else
        temp(i)=min(train_output);
    end
end
end_time_train = cputime;
Time = end_time_train - start_time_train;

TN=0;
TP=0;
FN=0;
FP=0;
for i=1:size(temp)
    if (temp(i)==train_output(i))& (temp(i)==0)
        TP=TP+1;
    end
    if (temp(i)==train_output(i))& (temp(i)==1)
        TN=TN+1;
    end
    if (train_output(i)==0)& (temp(i)==1)
        FN=FN+1;
    end
    if (train_output(i)==1)& (temp(i)==0)
        FP=FP+1;
    end
end
Accuracy=(TN+TP)/(TN+TP+FN+FP);

Recall=TP/(TP+FN);

Precision=TP/(TP+FP);

% fprintf('==================================================\n');
% fprintf('Run_Train_FIS.m done. Running Run_Test_FIS.m...  \n');
% fprintf('==================================================\n');

% addpath('..\output');
% save(['..\output\FIS_para.mat'], 'sigma_M');
