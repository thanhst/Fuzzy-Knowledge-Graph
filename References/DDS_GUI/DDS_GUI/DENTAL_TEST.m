clear;
clc;
load data;
load RuleList;
load FIS_defuzz;
load FIS_para;
rule_num=size(ruleList,1);
data_num=size(test_input,1);
attri_num=size(test_input,2);
term_num=3;
defuzz_num=5;
output_order=6;

degree_M=zeros(data_num,attri_num,term_num);
D_err_lam = zeros(data_num,rule_num);
exp_M=zeros(data_num,attri_num,term_num);
D_err_defuzzM=zeros(data_num,term_num);


j_M=0;
for i=1:data_num
    for j=1:attri_num
        for k=1:term_num
            exp_M(i,j,k)=gaussmf(test_input(i,j),[sigma_M(j,k) centers(j,k)]);
            degree_M(i,j,k) = exp_M(i,j,k);
        end   
    end
end
min_M=1;
M_DataPerRule = zeros(data_num,rule_num);
for i=1:data_num
    for j=1:rule_num
        min_M=1;
        for k=1:attri_num
            if min_M>degree_M(i,k,ruleList(j,k));
                min_M=degree_M(i,k,ruleList(j,k));
            end
        end
        M_DataPerRule(i,j)=min_M;
    end 
end
temp_M=zeros(data_num,rule_num);
result=zeros(data_num,1);
wtsum=zeros(data_num,1);

for i=1:data_num
    result(i)=0;
    wtsum(i)=0;
    for j=1:rule_num
        wtsum(i)=wtsum(i)+M_DataPerRule(i,j); 
        result(i)=result(i)+M_DataPerRule(i,j)*defuzz_M(ruleList(j,output_order)); 
    end
    result(i)=result(i)/wtsum(i);
end

temp=round(result);
for i=1:data_num
    if temp(i)<1
        temp(i)=1;
    else if temp(i)>5
            temp(i)=5;
        end
    end
end

% compare=zeros(5,5);
% for i=1:data_num
%     compare(temp(i),test_output(i))=compare(temp(i),test_output(i))+1;
% end
% numerator=0;
% for i=1:5
%     numerator=numerator+compare(i,i);
% end
% denominator=sum(sum(compare));
% 
% %Accuracy
% AccuracyTesting = num2str((numerator/denominator),'%0.5f');
% 
% %MAE
% MAETesting= num2str(mae(temp-test_output),'%0.5f');
% 
% %MSE
% MSETesting= num2str(mse(temp-test_output),'%0.5f');

% clc;
% load ValidityTraining.mat;
% fprintf('\t\t\t\t\t\t\t======== FUZZY INFERENCE SYSTEM ========\n'); 
% fprintf('+---------------+---------------+---------------+---------------+---------------+---------------+\n');
% fprintf('|\t\t\t\t\tTraining\t\t\t\t\t|\t\t\t\t\tTesting\t\t\t\t\t\t|\n');
% fprintf('+---------------+---------------+---------------+---------------+---------------+---------------+\n');
% fprintf('|\tAccuracy\t|\tMAE\t\t\t|\tMSE\t\t\t|\tAccuracy\t|\tMAE\t\t\t|\tMSE\t\t\t|\n');
% fprintf('|\t%s\t\t|\t%s\t\t|\t%s\t\t|\t%s\t\t|\t%s\t\t|\t%s\t\t|\n',...
%     AccuracyTraining,MAETraining,MSETraining,AccuracyTesting,MAETesting,MSETesting);
% fprintf('+---------------+---------------+---------------+---------------+---------------+---------------+\n');

save 'Result.mat' temp

delete RuleList.mat
delete FIS_defuzz.mat
delete FIS_para.mat;
delete data.mat;
delete FIS_fval.mat;
delete ValidityTraining.mat;