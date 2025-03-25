%Complex weighted error
clear;
clc;
%learing rate
eta=0.000001;
output_order=6;
term_num=3;

load FIS_para.mat;
load 'FIS_defuzz.mat';
load 'RuleList.mat';
load 'data.mat';

input_num=size(train_input,1);
attri_num=size(train_input,2);
rule_num=size(ruleList,1);
degree_M=zeros(input_num,attri_num,term_num);
Di_muy_submuy=zeros(input_num,rule_num);
wtsum = zeros(input_num,1);

d_center=zeros(attri_num,term_num);
d_sigma=zeros(attri_num,term_num);

D_err_muy=zeros(input_num,rule_num);
D_submuy_center=zeros(input_num,rule_num,attri_num);
D_submuy_sigma=zeros(input_num,rule_num,attri_num);
M_DataPerRule = zeros(input_num,rule_num);

iter_num=200;
FIS_fval2=zeros(iter_num,1);
for t=1:iter_num
    for i=1:input_num
        for j=1:attri_num
            for k=1:term_num
                degree_M(i,j,k) = gaussmf(train_input(i,j),[sigma_M(j,k) centers(j,k)]);
            end   
        end
    end
    min_M=1;

    for i=1:input_num
        for j=1:rule_num
            min_M=1;
            for k=1:attri_num
                if min_M>degree_M(i,k,ruleList(j,k));
                    min_M=degree_M(i,k,ruleList(j,k));
                    Di_muy_submuy(i,j)=k;
                end
            end
            M_DataPerRule(i,j)=min_M;
        end 
    end
    result=zeros(input_num,1);
    sum_M =zeros(input_num,1);
    for i=1:input_num
        result(i)=0;wtsum(i)=0;
        for j=1:rule_num
            wtsum(i)=wtsum(i)+M_DataPerRule(i,j);
            result(i)=result(i)+M_DataPerRule(i,j)*defuzz_M(ruleList(j,output_order));
        end
        result(i)=result(i)/wtsum(i);
    end
    error=result-train_output;
    err=sqrt(mse(error));
    FIS_fval2(t)=err;
    ['round ' num2str(t) ' : ' num2str(err)]
    for i=1:input_num
        for r=1:rule_num
            D_err_muy(i,r)=error(i)/input_num*(defuzz_M(ruleList(r,output_order))-result(i))/wtsum(i);
        end
    end
    for i=1:input_num
        for r=1:rule_num
            for j=1:attri_num
                D_submuy_center(i,r,j)=degree_M(i,j,ruleList(r,j))*(train_input(i,j)-centers(j,ruleList(r,j)))/power(sigma_M(j,ruleList(r,j)),2);
                D_submuy_sigma(i,r,j)=degree_M(i,j,ruleList(r,j))*(train_input(i,j)-centers(j,ruleList(r,j)))^2/power(sigma_M(j,ruleList(r,j)),3);
            end
        end
    end
    d_center=zeros(attri_num,term_num);
    d_sigma=zeros(attri_num,term_num);
    for i=1:input_num
        for r=1:rule_num
            j=Di_muy_submuy(i,r);
            d_center(j,ruleList(r,j))=d_center(j,ruleList(r,j))+D_err_muy(i,r)*D_submuy_center(i,r,j);
            d_sigma(j,ruleList(r,j))=d_sigma(j,ruleList(r,j))+D_err_muy(i,r)*D_submuy_sigma(i,r,j);
        end
    end
    centers=centers-eta*d_center;
    sigma_M=sigma_M-eta*d_sigma;
end

temp=round(result);
data_num=size(result,1);
for i=1:data_num
    if temp(i)<1
        temp(i)=1;
    else if temp(i)>5
            temp(i)=5;
        end
    end
end
compare=zeros(5,5);
for i=1:data_num
    compare(temp(i),train_output(i))=compare(temp(i),train_output(i))+1;
end
numerator=0;
for i=1:5
    numerator=numerator+compare(i,i);
end
denominator=sum(sum(compare));

%Accuracy
AccuracyTraining = num2str((numerator/denominator),'%0.5f');

%MAE
MAETraining=num2str(mae(temp-train_output),'%0.5f');

%MSE
MSETraining=num2str(mse(temp-train_output),'%0.5f');
% clc;
% fprintf('==============================================\n');
% fprintf('  DENTAL_FIS2.m done. Run DENTAL_TEST.m next.  \n');
% fprintf('==============================================\n');

% fprintf('\t\t\t======== RESULT ========\n'); 
% fprintf('+---------------+---------------+---------------+\n');
% fprintf('|\t\t\t\t\tTraining\t\t\t\t\t|\n');
% fprintf('+---------------+---------------+---------------+\n');
% fprintf('|\tAccuracy\t|\tMAE\t\t\t|\tMSE\t\t\t|\n');
% fprintf('|\t%s\t\t|\t%s\t\t|\t%s\t\t|\n',AccuracyTraining,MAETraining,MSETraining);
% fprintf('+---------------+---------------+---------------+\n');

save 'FIS_fval.mat' FIS_fval2 -append;
save 'FIS_para.mat' sigma_M;
save 'FIS_para.mat' centers -append;
save 'ValidityTraining.mat' AccuracyTraining;
save 'ValidityTraining.mat' MAETraining -append;
save 'ValidityTraining.mat' MSETraining -append;