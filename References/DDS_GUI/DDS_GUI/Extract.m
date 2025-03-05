function [ features ] = Extract( select )
%EXTRACT Summary of this function goes here
%   Detailed explanation goes here
    save select.mat select;
    FeatureExtraction;
    load resultFeature.mat;
    features=resultFeature;
end