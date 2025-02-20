% FCM_Fuction returns center V, Fuzzy Partition Matrix U,
% and Objective Funtion J

% Parameter:
% 1. X: data
% 2. C: number of cluster
% 3. V: initial centers
% 4. m, eps, maxTest: optional parameters

% Usage:
% [V,U,J] = FCM_Function(X,C,V,m,eps,maxTest)

function [V,U,cum]=FCM_Function(X,C,V,m,eps,maxTest)
    [N,r]=size(X);
    count = 0;
    while (1>0)
        for i = 1:N
            for j = 1:C
                Nominator = 0;
                for k = 1:r
                    Nominator = Nominator + power(X(i,k)-V(j,k),2);
                end    
                Nominator = sqrt(Nominator);
                Sum = 0;
                for l =1:C
                    Denominator = 0;
                    for k = 1:r
                        Denominator = Denominator + power(X(i,k)-V(l,k),2);
                    end
                    Denominator = sqrt(Denominator);
                    if Denominator~=0
                        Sum = Sum + power(Nominator/Denominator, 2/(m-1));
                    else
                        Sum=Sum;
                    end
                end
                if Sum==0
                    U(j,i)=1;
                else
                U(j,i) = 1/Sum;
                end
            end
        end
        % Normalize U
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
        % Compute W from U
        for j = 1:C
            for i = 1:r
                W_nominator = 0;
                W_denominator = 0;
                for k = 1:N
                    W_denominator = W_denominator + power(U(j,k),m);
                    W_nominator = W_nominator + power(U(j,k),m) * X(k,i);
                end
                if (W_denominator ~= 0) 
                    W(j,i) = W_nominator / W_denominator;
                else
                    W(j,i) = 0;
                end
            end
        end
        diff = 0;
        for i = 1:C
            for j = 1:r
                diff = diff + power(W(i,j)-V(i,j),2);
            end
        end
        diff = sqrt(diff);
    
        if (diff <= eps)
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
    % Compute J
    % Cluster
    maxU=max(U);
    for i=1:N
        for j=1:C
            if U(j,i)==maxU(i)
                cum(i)=j;
            end
        end
    end
     
    J = 0;
    for k=1:N
        Sum1 = 0;
        for j=1:C
            Sum2 = 0;
            for i=1:r
                Sum2 = Sum2 + power(X(k,i)-V(j,i),2);
            end
            Sum1 = Sum1 + power(Sum2,1-m);
        end
        J = J + power(Sum1,1-m);
    end
end