                            sum_sq     df            F        PR(>F)
C(method)             45113.524135    4.0   316.660902  7.806878e-56
C(metric)            183706.879483    4.0  1289.475547  4.557962e-85
C(method):C(metric)  136099.301137   16.0   238.827094  4.722053e-72
Residual               3561.658844  100.0          NaN           NaN
    
                         Multiple Comparison of Means - Tukey HSD, FWER=0.05
======================================================================================================
           group1                         group2               meandiff p-adj   lower    upper  reject
------------------------------------------------------------------------------------------------------
          Feature selection        Filter multimodal selection   7.3902 0.9869 -33.2761 48.0564  False
          Feature selection                 Hadamard selection  20.5554 0.6289 -20.1109 61.2216  False
          Feature selection                   Tensor selection -24.1015 0.4741 -64.7677 16.5648  False
          Feature selection Wrapper-based Multimodal selection  -29.662 0.2627 -70.3283 11.0042  False
Filter multimodal selection                 Hadamard selection  13.1652 0.8976 -27.5011 53.8315  False
Filter multimodal selection                   Tensor selection -31.4916 0.2082 -72.1579  9.1747  False
Filter multimodal selection Wrapper-based Multimodal selection -37.0522 0.0922 -77.7185  3.6141  False
         Hadamard selection                   Tensor selection -44.6568 0.0237 -85.3231 -3.9905   True
         Hadamard selection Wrapper-based Multimodal selection -50.2174 0.0075 -90.8837 -9.5511   True
           Tensor selection Wrapper-based Multimodal selection  -5.5606 0.9956 -46.2269 35.1057  False
------------------------------------------------------------------------------------------------------