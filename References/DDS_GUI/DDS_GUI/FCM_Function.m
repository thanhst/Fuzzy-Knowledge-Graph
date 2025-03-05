% FCM_Fuction returns center V, Fuzzy Partition Matrix U,
% and Objective Funtion J

% Parameter:
% 1. X: data
% 2. C: number of cluster
% 3. V: initial centers
% 4. m, eps, maxTest: optional parameters

% Usage:
% [V,U,J] = FCM_Function(X,C,V,m,eps,maxTest)

function [V,U,J]=FCM_Function(X,C,V,m,eps,maxTest)
    [N,r]=size(X);
    count = 0;
    while (1>0)
    for i = 1:N
           for j = 1:C
              Tu = 0;
              for k = 1:r
                  Tu = Tu + power(X(i,k)-V(j,k),2);
              end    
              Tu = sqrt(Tu);

              tong = 0;
              for l =1:C
                  Mau = 0;
                  for k = 1:r
                      Mau = Mau + power(X(i,k)-V(l,k),2);
                  end    

                  Mau = sqrt(Mau);
                  if Mau~=0
                      tong = tong + power(Tu/Mau, 2/(m-1)); 
                  else
                      tong=tong;
                  end
              end
            if tong==0
                U(j,i)=1;
            else
              U(j,i) = 1/tong;  
            end    

           end
    end
     maxU=max(U);
     for i=1:N
         if maxU(i)==1
             for j=1:C
                 if U(j,i)~=maxU(i)
                     U(j,i)=0;
                 end
             end
         end
     end

     U 
     %chuan hoa U


            %Tinh V(t+1) tu U ra W
	for j = 1:C
        for i = 1:r
            tuso = 0;
            mauso = 0;
            for k = 1:N
                mauso = mauso + power(U(j,k),m);
            	tuso = tuso + power(U(j,k),m) * X(k,i);
            end
            if (mauso ~= 0) 
            	W(j,i) = tuso / mauso;
            else
            	W(j,i) = 0;
            end
        end
    end   
       
	saiso = 0;
	for i = 1:C
        for j = 1:r              
            saiso = saiso + power(W(i,j)-V(i,j),2); 
        end
    end
    saiso = sqrt(saiso);

	if (saiso <= eps)
        break;
    else          
        for i = 1:C
            for j = 1:r
                V(i,j)= W(i,j);
            end
        end
    end
 
	if(count>=maxTest)
        break;
    end
        count = count + 1;
    end

	J=0;
    for k=1:N
        tong1=0;
        for j=1:C
            tong2=0;
            for i=1: r
                tong2=tong2+power(X(k,i)-V(j,i),2);
            end
            tong1=tong1+power(tong2,1-m);
        end
        J=J+power(tong1,1-m);
    end
	maxU = max(U)
	for i = 1 : C
        for j = 1:N
            if U(i,j)==maxU(j)
                cum(j)=i;
            end

        end
    end
end

function t = calcSumDistDataPoint2X(data, X)
    temp = data - X(ones(size(data, 1), 1), :);
    temp = temp.^2;
    temp = sum(temp, 2);
    temp = sqrt(temp);
    t = sum(temp);
end


function PBM_value = PBM(numClust, data, center, U)
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
end


function SWC_value = SWC(numClust, data, U)
    maxU = max(U);
    SWC_value = 0;

    wb = waitbar(0);
    count = 0;

    for i = 1 : numClust
        index = find(U(i,:) == maxU);
        if size(index, 2) == 1
            continue;
        end
        clustData = data(index, :);
        for j = index
            a_i_j = calcSumDistDataPoint2X(clustData, data(j, :)) / size(index, 2);
            b_i_j = inf;
            for k = 1 : numClust
                if k ~= i
                    index_k = find(U(k,:) == maxU);
                    clustData_k = data(index_k, :);
                    d_k_j = calcSumDistDataPoint2X(clustData_k, data(j, :)) / size(index_k, 2);
                    b_i_j = min(b_i_j, d_k_j);
                end
            end
            SWC_value = SWC_value + (b_i_j - a_i_j) / max(a_i_j, b_i_j);
            count = count + 1;

    %         waitbar(count / size(data, 1), wb);
        end
    end
    SWC_value = SWC_value / size(data, 1);
end


        
function DB_value = DB(numClust, data, center, U)
    maxU = max(U);
    
    for i = 1 : numClust
        index = find(U(i,:) == maxU);
        clustData = data(index, :);
        stdData = std(clustData, 1);
        S(i) = sqrt(stdData(1)^2 + stdData(2)^2 + stdData(3)^2);
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
end

    
function IFV_value = IFV(numClust, data, sizeData, center, U)
    sigmaD = 0;
    sum = 0;
    
    for i = 1:numClust
        tg1 = 0;
        tg2 = 0;
        for j = 1:sizeData
            if U(j,i) == 0 
                U(j, i) = eps;
            end
            if U(j, i) == 1 
                U(j, i) = 1 - eps;
            end
            
            tg1 = tg1 + log(U(j, i))/log(2);
            tg2 = tg2 + U(j, i)^2;
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
end