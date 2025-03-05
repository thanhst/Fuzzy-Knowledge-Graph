function [ disease ] = Analyze(select)
%ANALYZE Summary of this function goes here

%   Detailed explanation goes here

    %Start of master_function.m file
    
    save select.mat select;
    waitbar(0.1);
    diary log1.txt;FeatureExtraction;
    diary log1.txt;RUN_THIS_FIRST_FIS;
    diary log2.txt;MakeData;
    waitbar(0.25);
    diary log3.txt;DENTAL_FIS;
    waitbar(0.5);
    diary log4.txt;DENTAL_FIS2;
    waitbar(0.85);
    diary log5.txt;DENTAL_TEST;
    pause(1);
    waitbar(1);
    pause(1);
    %End of master_function.m file

    load Result.mat;
    if(isnan(temp))
        temp=0;
    end
    disease=temp;
    delete Result.mat;
    delete select.mat;
    delete resultFeature.mat;
end

