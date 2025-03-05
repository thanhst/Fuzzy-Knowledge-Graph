% FeatureExtraction returns a set of five Dental Features of X-Ray Images.
% Five features will be extracted including 
% 1. Entropy - Edge - Intensity (EEI)
% 2. Local Binary Pattern (LBP)
% 3. Red - Green - Blue (RGB)
% 4. Gradient
% 5. Patch
% Each feature is caculated by mean of the value.
%
% This script scans all image file in a folder, extracts five features,
% and save to a file. A record in text file which corresponds to an image
% has ID, EEI, LBP, RGB, Gradient, Patch, respectively. (ID is an idenity
% of an image). 
%
% Label retrieved from expert's advice, and marked as follow:
% 1.  
%
% Usage: put this script into the same folder which contains images.

% READ IMAGE
% clear;
% timer;
% Val=tic;
% tic;
% srcFiles = dir('*.jpg');  % the folder in which ur images exists

% for id = 1 : length(srcFiles)
%     filename = strcat('',srcFiles(id).name);
    load select.mat;
    im=select;


    imgray=rgb2gray(im);

    %===============================
    %  Entropy - Edge - Intensity  |
    %===============================

    % Entropy
    E=entropy(imgray);

    % Edge
    BW = edge(imgray);
    h=size(im,1);
    w=size(im,2);
    S=size(im,1)*size(im,2);
    S1=0;
    for i=1:h
        for j=1:w
            if(BW(i,j)==1)
                S1=S1+1;
            end
        end    
    end
    avgBW=S1/S;

    % Intensity
    Intensity=imadjust(imgray);
    avgIntensity=mean2(Intensity);

    % Mean of EEI
    resultEnEdIn=(E+avgBW+avgIntensity)/3;

    %==========================
    %  Local Binary Patterns  |
    %==========================
    h=size(imgray,1);
    w=size(imgray,2);

    %upper-left
    LBP(1,1)=(imgray(1,2)>=imgray(1,1))*2^4+(imgray(2,2)>=imgray(1,1))*2^3+(imgray(2,1)>=imgray(1,1))*2^2;
    %upper-right
    LBP(1,w)=(imgray(1,w-1)>=imgray(1,1))*2^0+(imgray(2,w-1)>=imgray(1,1))*2^1+(imgray(2,w)>=imgray(1,1))*2^2;
    %lower-left
    LBP(h,1)=(imgray(h-1,1)>=imgray(1,1))*2^6+(imgray(h-1,2)>=imgray(1,1))*2^5+(imgray(h,2)>=imgray(1,1))*2^4;
    %lower-right
    LBP(h,w)=(imgray(h-1,w-1)>=imgray(1,1))*2^7+(imgray(h-1,w)>=imgray(1,1))*2^6+(imgray(h,w-1)>=imgray(1,1))*2^0;

    %top border
    for j=2:w-1
        LBP(1,j)=(imgray(1,j-1)>=imgray(1,j))*2^0+(imgray(2,j-1)>=imgray(1,j))*2^1+(imgray(2,j)>=imgray(1,j))*2^2+(imgray(1,j+1)>=imgray(1,j))*2^3+(imgray(1,j+1)>=imgray(1,j))*2^4;
    end

    %bottom border
    for j=2:w-1
        LBP(h,j)=(imgray(h-1,j-1)>=imgray(h,j))*2^7+(imgray(h-1,j)>=imgray(h,j))*2^6+(imgray(h-1,j+1)>=imgray(h,j))*2^5+(imgray(h,j+1)>=imgray(h,j))*2^4+(imgray(h,j-1)>=imgray(h,j))*2^0;
    end

    %left border
    for i=2:h-1
        LBP(i,1)=(imgray(i-1,1)>=imgray(i,1))*2^6+(imgray(i-1,2)>=imgray(i,1))*2^5+(imgray(i,2)>=imgray(i,1))*2^4+(imgray(i+1,2)>=imgray(i,1))*2^3+(imgray(i+1,1)>=imgray(i,1))*2^2;
    end

    %right border
    for i=2:h-1
        LBP(i,w)=(imgray(i-1,w-1)>=imgray(i,w))*2^7+(imgray(i-1,w)>=imgray(i,w))*2^6+(imgray(i+1,w)>=imgray(i,w))*2^2+(imgray(i+1,w-1)>=imgray(i,w))*2^1+(imgray(i,w-1)>=imgray(i,w))*2^0;
    end

    %inside
    for i=2:h-1
        for j=2:w-1
            LBP(i,j)=(imgray(i-1,j-1)>=imgray(i,j))*2^7+(imgray(i-1,j)>=imgray(i,j))*2^6+(imgray(i-1,j+1)>=imgray(i,j))*2^5+(imgray(i,j+1)>=imgray(i,j))*2^4+(imgray(i+1,j+1)>=imgray(i,j))*2^3+(imgray(i+1,j)>=imgray(i,j))*2^2+(imgray(i+1,j-1)>=im(i,j))*2^1+(imgray(i,j-1)>=imgray(i,j))*2^0;
        end
    end

    %mean of LBP
    resultLBP=mean2(LBP);

    %=======================
    %  Red - Green - Blue  |
    %=======================

    h=size(im,1);
    w=size(im,2);
    N=h*w;

    avgR=mean2(im(:,:,1));
    avgG=mean2(im(:,:,2));
    avgB=mean2(im(:,:,3));

    resultRGB=(avgR+avgG+avgB)/3;

    %=============
    %  Gradient  |
    %=============
    [m1,n,c]=size(im);
    R1 = im(:, : , 1);
    G1 = im(:, : , 2);
    B1 = im(:, : , 3);
    clearvars D;
    for i=1:m1
       D(i,:)= gradient(R1(i,:),G1(i,:),B1(i,:));
    end
    resultGradient=mean2(D);

    %==========
    %  Patch  |
    %==========
    R = im(:, : , 1);
    G = im(:, : , 2);
    B = im(:, : , 3);
    data = double ([R(:) G(:) B(:)]);
    resultPatch=patch(R,G,B,'b');
    %close;

    % WRITE TO A FILE
    resultFeature=[resultEnEdIn,resultLBP,resultRGB,resultGradient,resultPatch];
    dlmwrite('features_log.txt',resultFeature,'-append')
    save resultFeature.mat resultFeature;
% end
% toc;