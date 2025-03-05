function value = ValidityIndex(indexName, data, numClust, center, U)
    switch indexName
        case 'DB'
            value = DB(data, numClust, center, U);
        case 'IFV'
            value = IFV(data, numClust, center, U);
        case 'PBM'
            value = PBM(data, numClust, center, U);
        case 'SWC'
            value = SWC(data, numClust, U);
        otherwise
            value = nan;
    end

function DB_value = DB(data, numClust, center, U)
    maxU = max(U);
    
    S = zeros(1, numClust);
    
    for i = 1 : numClust
        index = find(U(i,:) == maxU);
        
        for j = index
            S(i) = S(i) + norm(data(j, :) - center(i, :)) ^ 2;
        end
        S(i) = sqrt(S(i)/size(index, 2));
    end
    
    DB_value = 0;
    
    for i = 1:numClust
        maxSM = 0;
        for j = 1:numClust
           if j ~= i
               temp = (S(i) + S(j))/norm(center(i, :) - center(j, :));
               maxSM = max(maxSM, temp);
           end
        end
        DB_value = DB_value + maxSM;
    end
    
    DB_value = DB_value/numClust;

    
    
function IFV_value = IFV(data, numClust, center, U)
    sigmaD = 0;
    sum = 0;
    sizeData = size(data,1);
    
    for i = 1:numClust
        tg1 = 0;
        tg2 = 0;
        for j = 1:sizeData
%             if U(i, j) == 0 
%                 U(i, j) = eps;
%             end
%             if U(i, j) == 1 
%                 U(i, j) = 1 - eps;
%             end
            
            tg1 = tg1 + log(U(i, j))/log(2);
            tg2 = tg2 + U(i, j)^2;
            sigmaD = sigmaD + norm(data(j, :) - center(i, :))^2;
        end
        
        tg = (log(numClust)/log(2) - tg1/sizeData)^2;
        tg2 = tg2/sizeData;
        
        sum = sum + tg * tg2;
    end
    
    sigmaD = sigmaD/(numClust * sizeData);
    
    calcSDmax = 0;
    for i = 1:numClust-1
        for j = i+1:numClust
            calcSDmax = max(calcSDmax, norm(center(i, :) - center(j, :))^2);
        end
    end
    
    IFV_value  = (sum * calcSDmax) / (sigmaD * numClust);
    

  % calculate sum of distances between each data point in data and X
function t = calcSumDistDataPoint2X(data, X)
temp = data - X(ones(size(data, 1), 1), :);
temp = temp.^2;
temp = sum(temp, 2);
temp = sqrt(temp);
t = sum(temp);


function PBM_value = PBM(data, numClust, center, U)
E_1 = calcSumDistDataPoint2X(data, mean(data));

maxU = max(U);
E_k = 0;

for i = 1 : numClust
    index = find(U(i,:) == maxU);
    clustData = data(index, :);
    E_k = E_k +  calcSumDistDataPoint2X(clustData, center(i, :));
end

D_k = 0;
for i = 1:numClust-1
    for j = i+1:numClust
        D_k = max(D_k, norm(center(i, :) - center(j, :)));
    end
end
    
PBM_value = (E_1 * D_k / (numClust * E_k)) ^ 2;


function SWC_value = SWC(data, numClust, U)
maxU = max(U);
SWC_value = 0;

for i = 1 : numClust
    index = find(U(i,:) == maxU);
    
    if size(index, 2) == 1
        continue;
    end
    
    clustData = data(index, :);
    
    for j = index
        a_i_j = calcSumDistDataPoint2X(clustData, data(j, :)) / size(index, 2);
        b_i_j = 10^6;
        
        for k = 1 : numClust
            if k ~= i
                index_k = find(U(k,:) == maxU);
                
                clustData_k = data(index_k, :);
                
                d_k_j = calcSumDistDataPoint2X(clustData_k, data(j, :)) / size(index_k, 2);
                b_i_j = min(b_i_j, d_k_j);
            end
        end
        SWC_value = SWC_value + (b_i_j - a_i_j) / max(a_i_j, b_i_j);
    end
end
SWC_value = SWC_value / size(data, 1);
