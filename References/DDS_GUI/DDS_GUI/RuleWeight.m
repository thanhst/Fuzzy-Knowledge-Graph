% RuleWeight returns matrix t, and sigma
% matrix t is used to determine weight of each rule
% sigma is for gauss function is this function and later use in FIS

% Parameter:
% 1. rules: rules created before
% 2. data: dental data to generate rule before
% 3. cluster: number of cluster for each feature
% 4-8. center 1 to 6: center retrieved from FCM_Function

% Usage:
% [t,sigma] = RuleWeight (rules, data,cluster, center1, center2,...
%                center3, center4, center5, center6)


function [t,sigma] = RuleWeight (rules, data,cluster, center1, center2,...
                center3, center4, center5, center6)
    [height,width]=size(data);
    
    %Feature 1 Entropy
    feature1=data(:,2);
    rule1=rules(:,1);
    MFnumber=cluster(1,1);
    sigma1=(sqrt((center1(1,1)-center1(2,1))^2))/(2*sqrt(2*log(2)));
    for i=1:height
        t(i,1)=GaussMF(feature1(i,1),rule1(i,1),MFnumber,sigma1,center1);
    end
    
    %Feature 2 LBP
    feature2=data(:,3);
    rule2=rules(:,2);
    MFnumber=cluster(1,2);
    sigma2=(sqrt((center2(1,1)-center2(2,1))^2))/(2*sqrt(2*log(2)));
    for i=1:height
        t(i,2)=GaussMF(feature2(i,1),rule2(i,1),MFnumber,sigma2,center2);
    end
    
    %Feature 3 RGB
    feature3=data(:,4);
    rule3=rules(:,3);
    MFnumber=cluster(1,3);
    sigma3=(sqrt((center3(1,1)-center3(2,1))^2))/(2*sqrt(2*log(2)));
    for i=1:height
        t(i,3)=GaussMF(feature3(i,1),rule3(i,1),MFnumber,sigma3,center3);
    end
    
    %Feature 4 Gradient
    feature4=data(:,5);
    rule4=rules(:,4);
    MFnumber=cluster(1,4);
    sigma4=(sqrt((center4(1,1)-center4(2,1))^2))/(2*sqrt(2*log(2)));
    for i=1:height
        t(i,4)=GaussMF(feature4(i,1),rule4(i,1),MFnumber,sigma4,center4);
    end
    
    %Feature 5 Patch
    feature5=data(:,6);
    rule5=rules(:,5);
    MFnumber=cluster(1,5);
    sigma5=(sqrt((center5(1,1)-center5(2,1))^2))/(2*sqrt(2*log(2)));
    for i=1:height
        t(i,5)=GaussMF(feature5(i,1),rule5(i,1),MFnumber,sigma5,center5);
    end
    
    %Label - Dental Problem
    label=data(:,7);
    rule6=rules(:,6);
    MFnumber=cluster(1,6);
    sigma6=(sqrt((center6(1,1)-center6(2,1))^2))/(2*sqrt(2*log(2)));
    for i=1:height
        t(i,6)=GaussMF(label(i,1),rule6(i,1),MFnumber,sigma6,center6);
    end
    
   sigma=[sigma1 sigma2 sigma3 sigma4 sigma5 sigma6];
   sigma=sigma';
end

%Gauss membership function
function [ y ] = GaussMF( x1, label, MFnumber, sigma, center )
    switch MFnumber
        case 2
            if(label==1)
                y = gaussmf(x1,[sigma center(1,1)]);
            elseif (label==2)
                y = gaussmf(x1,[sigma center(2,1)]);
            end
        case 3            
            if(label==1)
                y = gaussmf(x1,[sigma center(1,1)]);
            elseif (label==2)
                y = gaussmf(x1,[sigma center(2,1)]);
            elseif (label==3)
                y = gaussmf(x1,[sigma center(3,1)]);
            end
        case 4
            if(label==1)
                y = gaussmf(x1,[sigma center(1,1)]);
            elseif (label==2)
                y = gaussmf(x1,[sigma center(2,1)]);
            elseif (label==3)
                y = gaussmf(x1,[sigma center(3,1)]);
            elseif (label==4)
                y = gaussmf(x1,[sigma center(4,1)]);
            end
            case 5
            if(label==1)
                y = gaussmf(x1,[sigma center(1,1)]);
            elseif (label==2)
                y = gaussmf(x1,[sigma center(2,1)]);
            elseif (label==3)
                y = gaussmf(x1,[sigma center(3,1)]);
            elseif (label==4)
                y = gaussmf(x1,[sigma center(4,1)]);
            elseif (label==5)
                y = gaussmf(x1,[sigma center(5,1)]);
            end
            case 6
            if(label==1)
                y = gaussmf(x1,[sigma center(1,1)]);
            elseif (label==2)
                y = gaussmf(x1,[sigma center(2,1)]);
            elseif (label==3)
                y = gaussmf(x1,[sigma center(3,1)]);
            elseif (label==4)
                y = gaussmf(x1,[sigma center(4,1)]);
            elseif (label==5)
                y = gaussmf(x1,[sigma center(5,1)]);
            elseif (label==6)
                y = gaussmf(x1,[sigma center(6,1)]);    
            end
        otherwise
    end
end