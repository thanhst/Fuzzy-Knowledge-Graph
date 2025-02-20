% RuleWeight returns matrix t, and sigma
% matrix t is used to determine weight of each rule
% sigma is for gauss function is this function and later use in FIS

% Parameter:
% 1. rules: rules created before
% 2. data: data to generate rule before
% 3. cluster: number of cluster for each feature
% 4. center_vector: contains N centers retrieved from FCM_Function
% N is number of attributes include label

% Usage:
% [t,sigma] = RuleWeight (rules, data, cluster, center_vector)


function [t,sigma] = RuleWeight(rules, data,cluster, center_vector)
    [data_num attribute_num] = size(data);
    sigma = zeros(1,attribute_num);
    for feature_index=1:attribute_num
        feature_data = data(:,feature_index);
        rule_index = rules(:,feature_index);
        MFnumber = cluster(feature_index);
        sigma(feature_index) = compute_sigma(center_vector{feature_index});
        for i=1:data_num
            t(i,feature_index)=GaussMF(feature_data(i,1),rule_index(i,1),MFnumber,sigma(feature_index),center_vector{feature_index});
        end
    end
   sigma = sigma';
   
end

% Gauss membership function
function [ y ] = GaussMF( x1, label, MFnumber, sigma, center )
    for i=1:MFnumber
        if(label == i)
            y = gaussmf(x1,[sigma center(label,1)]);
            break
        end
    end
end

% Compute Sigma
function sigma = compute_sigma(center_vector)
    d = 0;
    if length(center_vector) == 2
        d = center_vector(1) - center_vector(2);
    else
        for i=1:length(center_vector)-1
            for j=i+1:length(center_vector)
                d_temp = center_vector(i) - center_vector(j);
                if abs(d_temp) > abs(d)
                    d = d_temp;
                end
            end
        end
    end
    sigma = abs(d)/(2*sqrt(2*log(2)));
    while sigma < 1
        sigma = sigma*10;
    end
end
