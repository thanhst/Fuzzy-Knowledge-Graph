import pandas as pd
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage import io, color, img_as_ubyte
import os
import cv2
import cupy as cp  # Sử dụng cupy thay vì numpy
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.utils import shuffle
from pathlib import Path
import time
import gc,csv
base_path = Path(__file__).resolve().parents[2]

# # Brightness and contrast enhancement function using CLAHE )
# def apply_clahe(image):

#     lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
#     l, a, b = cv2.split(lab)

#     clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
#     cl = clahe.apply(l)

#     # Merge lại các kênh
#     enhanced_img = cv2.merge((cl, a, b))
#     enhanced_img = cv2.cvtColor(enhanced_img, cv2.COLOR_LAB2BGR)
    
#     return enhanced_img

# # Hàm làm nét ảnh (Unsharp Masking) sử dụng
# def apply_unsharp_mask(image, amount=1.5, threshold=0):
#     blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=3)
#     sharpened = cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)
#     return sharpened # Download result back to CPU

# # Hàm cải thiện độ sáng với linear contrast stretching (dùng cupy)
# def linear_contrast_stretch(image):
#     image_gpu = cp.asarray(image)
    
#     min_val = cp.percentile(image_gpu, 2)
#     max_val = cp.percentile(image_gpu, 98)
    
#     stretched_gpu = cp.clip((image_gpu - min_val) * 255.0 / (max_val - min_val), 0, 255)
#     return cp.asnumpy(stretched_gpu).astype(np.uint8)  # Chuyển về numpy để tiếp tục xử lý

# # Hàm segment ảnh bằng KMeans (dùng CPU vì hiện tại KMeans chưa hỗ trợ GPU)
# def segment_by_kmeans(image, k=2):
#     Z = image.reshape((-1, 3))
#     Z = np.float32(Z)
#     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
#     ret, label, center = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
#     center = np.uint8(center)
#     res = center[label.flatten()]
#     segmented_image = res.reshape((image.shape))

#     # Lấy mask của cluster tối nhất (vì tổn thương hay mảng da bất thường thường có màu sẫm)
#     darkest_cluster_idx = np.argmin(np.sum(center, axis=1))  # Tổng RGB thấp nhất
#     mask = (label.flatten() == darkest_cluster_idx).astype(np.uint8)
#     mask = mask.reshape((image.shape[0], image.shape[1]))

#     return segmented_image, mask

# def segment_by_otsu(gray_image):
#     # Áp dụng Gaussian Blur để giảm nhiễu trước khi phân ngưỡng
#     blur = cv2.GaussianBlur(gray_image, (5, 5), 0)
    
#     # Áp dụng Otsu Thresholding để phân đoạn ảnh
#     _, binary_mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
#     return binary_mask

# def segment_with_otsu_and_kmeans(image, k=2):
#     # Bước 1: Áp dụng Otsu Thresholding
#     gray = color.rgb2gray(image)
#     gray_image = img_as_ubyte(gray)
    
#     # Áp dụng Otsu Thresholding để phân đoạn ảnh
#     otsu_mask = segment_by_otsu(gray_image)
    
#     # Bước 2: Phân đoạn ảnh với KMeans
#     segmented_image, kmeans_mask = segment_by_kmeans(image, k)
    
#     # Bước 3: Kết hợp kết quả Otsu và KMeans
#     # Giữ lại vùng có giá trị lớn hơn ngưỡng Otsu và những vùng phân đoạn qua KMeans
#     combined_mask = otsu_mask & kmeans_mask  # Chỉ giữ lại các vùng mà cả Otsu và KMeans đều xác nhận
    
#     # Áp dụng mask để giữ lại các vùng quan trọng
#     masked_image = image.copy()
#     masked_image[combined_mask == 0] = 0  # Giữ lại những vùng không phải là nền
    
#     return masked_image, otsu_mask, kmeans_mask, combined_mask

# def preprocess_fundus_image(image):
#     # Làm nét
#     sharpened = apply_unsharp_mask(image)
    
#     denoised = cv2.fastNlMeansDenoisingColored(sharpened, None, 10, 10, 7, 21)
    
#     # Cải thiện độ tương phản bằng CLAHE
#     clahe_img = apply_clahe(denoised)

#     return clahe_img

# list_of_images = []
# path_of_images = os.path.join(base_path,"data/Image/fundus_photo/")
# images = os.listdir(path_of_images)
# list_of_images.extend([os.path.join(path_of_images, img) for img in images])

# total_time_preprocessing_image = 0
# total_time_segment_image = 0

# os.makedirs(os.path.join(base_path,"data/Dataset_diabetic/Fusion_feature"), exist_ok=True)


# def contrast_feature(matrix):
#     return np.mean(graycoprops(matrix, 'contrast'))

# def dissimilarity_feature(matrix):
#     return np.mean(graycoprops(matrix, 'dissimilarity'))

# def homogeneity_feature(matrix):
#     return np.mean(graycoprops(matrix, 'homogeneity'))

# def energy_feature(matrix):
#     return np.mean(graycoprops(matrix, 'energy'))

# def correlation_feature(matrix):
#     return np.mean(graycoprops(matrix, 'correlation'))

# def asm_feature(matrix):
#     return np.mean(graycoprops(matrix, 'ASM'))
    
# def mean_feature(matrix):
#     return np.mean(matrix)

# def variance_feature(matrix):
#     return np.var(matrix)

# def sd_feature(matrix):
#     return np.std(matrix)
    
# def rms_feature(matrix):
#     return np.sqrt(np.mean(np.square(matrix)))

# total_time_feature_selected_time = 0
# total_time_feature_extraction_time = 0

# feature_selected_time = time.time()
# if not os.path.exists(os.path.join(base_path,"data/Dataset_diabetic/Image_feature/images_ft.csv")):
#     with open(os.path.join(base_path,"data/Dataset_diabetic/Image_feature/images_ft.csv"), mode='w', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow([
#             "image_id",
#             "Contrast Feature",
#             "Dissimilarity Feature",
#             "Homogeneity Feature",
#             "Energy Feature",
#             "Correlation Feature",
#             "ASM Feature",
#             "Mean Feature",
#             "Variance Feature",
#             "Standard Deviation Feature",
#             "RMS Feature"
#         ])
# rs_feature_selected_time = time.time()
# total_time_feature_selected_time += rs_feature_selected_time-feature_selected_time

# for image in list_of_images:
#     time_preprocess_image = time.time()
#     img = cv2.imread(image)
#     img = cv2.resize(img, (256, 256))
#     img = preprocess_fundus_image(img)
#     segmented_img, otsu_mask, kmeans_mask, combined_mask = segment_with_otsu_and_kmeans(img, k=3)
#     total_time_preprocessing_image += time.time()-time_preprocess_image
    
#     time_segment_image = time.time()
#     gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # Bước 5: Áp dụng mask Otsu để giữ lại vùng tổn thương
#     masked_img = cv2.bitwise_and(gray_img, gray_img, mask=otsu_mask)

#     # Bước 6: Chuyển ảnh masked về dạng 8-bit nếu cần
#     masked_img_ubyte = img_as_ubyte(masked_img)

#     # Bước 7: Phân bin ảnh thành 16 mức xám

#     # Bước 8: Tính GLCM
#     matrix_coocurrence = graycomatrix(
#         masked_img_ubyte,
#         distances=[1],
#         angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
#         levels=256,
#         normed=True,
#         symmetric=True,
#     )
#     masked_inds = img[otsu_mask == 0]
#     CF = contrast_feature(matrix_coocurrence)
#     DF= dissimilarity_feature(matrix_coocurrence)
#     HF = homogeneity_feature(matrix_coocurrence)
#     EF = energy_feature(matrix_coocurrence)
#     COR = correlation_feature(matrix_coocurrence)
#     ASM = asm_feature(matrix_coocurrence)
#     MF = mean_feature(masked_inds)
#     VF = variance_feature(masked_inds)
#     SD = sd_feature(masked_inds)
#     RMS = rms_feature(masked_inds)
    
#     feature_extraction_time = time.time()

#     with open(os.path.join(base_path,"data/Dataset_diabetic/Image_feature/images_ft.csv"), mode='a', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow([
#             os.path.splitext(os.path.basename(image))[0],
#             CF,
#             DF,
#             HF,
#             EF,
#             COR,
#             ASM,
#             MF,
#             VF,
#             SD,
#             RMS
#         ])
#     rs_feature_extraction_time = time.time()
#     total_time_feature_extraction_time += rs_feature_extraction_time-feature_extraction_time
    
#     total_time_segment_image += time.time() - time_segment_image
#     del img, gray_img, masked_img, masked_img_ubyte, matrix_coocurrence,CF,DF,HF,EF,COR,ASM,MF,VF,SD,RMS
#     gc.collect()

# print(f'Total time processing image: {total_time_preprocessing_image}')
# print(f'Total time segment image: {total_time_segment_image}')

# df = pd.read_csv(os.path.join(base_path,"data/Dataset_diabetic/Image_Feature/images_ft.csv"))

# for col in df.columns[1:]:
#     df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    

# print(f'Feature extraction image total time: {total_time_feature_extraction_time}')


# print(f'Feature selected image total time: {total_time_feature_selected_time}')

# time_image = total_time_preprocessing_image + total_time_segment_image + total_time_feature_extraction_time + total_time_feature_selected_time
# print(f'Time processing image: {time_image}')

# dfData = pd.read_csv(os.path.join(base_path,"data/Dataset_diabetic/labels_brset.csv"))
# start_process_table = time.time()
# columns_to_keep = [
#     'image_id',
#     'diabetic_retinopathy'
# ]
# dfData = dfData[columns_to_keep]
# dfMerge = dfData.merge(df,how = 'inner', on= 'image_id')
# dfMerge = dfMerge.drop(['image_id'],axis=1)
# dfMerge = dfMerge[[col for col in dfMerge.columns if col != 'diabetic_retinopathy'] + ['diabetic_retinopathy']]

# os.makedirs(os.path.join(base_path,"data/Dataset_diabetic/Image_feature"), exist_ok=True)
# dfMerge.to_csv(os.path.join(base_path,"data/Dataset_diabetic/Image_feature/data_process.csv"),index=False)
# rs_feature_selected_time = time.time()
# total_time_feature_selected_time = rs_feature_selected_time-feature_selected_time
# print(f'Feature selected image total time: {total_time_feature_selected_time}')
# time_image = total_time_preprocessing_image + total_time_segment_image + total_time_feature_extraction_time + total_time_feature_selected_time
# print(f'Time processing image: {time_image}')

from sklearn.feature_selection import SelectKBest, mutual_info_classif
import pandas as pd
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage import io, color, img_as_ubyte
import os
import cv2
import cupy as cp  # Sử dụng cupy thay vì numpy
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.utils import shuffle
from pathlib import Path
import time
import gc
import csv
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

base_path = Path(__file__).resolve().parents[2]

def process_image(folder_save,file_path,k_img=7):
    # # Brightness and contrast enhancement function using CLAHE )
    def apply_clahe(image):

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)

        # Merge lại các kênh
        enhanced_img = cv2.merge((cl, a, b))
        enhanced_img = cv2.cvtColor(enhanced_img, cv2.COLOR_LAB2BGR)
        
        return enhanced_img

    # Hàm làm nét ảnh (Unsharp Masking) sử dụng
    def apply_unsharp_mask(image, amount=1.5, threshold=0):
        blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=3)
        sharpened = cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)
        
        return sharpened # Download result back to CPU

    # Hàm cải thiện độ sáng với linear contrast stretching (dùng cupy)
    def linear_contrast_stretch(image):
        image_gpu = cp.asarray(image)
        
        min_val = cp.percentile(image_gpu, 2)
        max_val = cp.percentile(image_gpu, 98)
        
        stretched_gpu = cp.clip((image_gpu - min_val) * 255.0 / (max_val - min_val), 0, 255)
        return cp.asnumpy(stretched_gpu).astype(np.uint8)  # Chuyển về numpy để tiếp tục xử lý

    # Hàm segment ảnh bằng KMeans (dùng CPU vì hiện tại KMeans chưa hỗ trợ GPU)
    def segment_by_kmeans(image, k=2):
        Z = image.reshape((-1, 3))
        Z = np.float32(Z)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret, label, center = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        res = center[label.flatten()]
        segmented_image = res.reshape((image.shape))

        # Lấy mask của cluster tối nhất (vì tổn thương hay mảng da bất thường thường có màu sẫm)
        darkest_cluster_idx = np.argmin(np.sum(center, axis=1))  # Tổng RGB thấp nhất
        mask = (label.flatten() == darkest_cluster_idx).astype(np.uint8)
        mask = mask.reshape((image.shape[0], image.shape[1]))

        return segmented_image, mask

    def segment_by_otsu(gray_image):
        # Áp dụng Gaussian Blur để giảm nhiễu trước khi phân ngưỡng
        blur = cv2.GaussianBlur(gray_image, (5, 5), 0)
        
        # Áp dụng Otsu Thresholding để phân đoạn ảnh
        _, binary_mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary_mask

    def segment_with_otsu_and_kmeans(image, k=2):
        # Bước 1: Áp dụng Otsu Thresholding
        gray = color.rgb2gray(image)
        gray_image = img_as_ubyte(gray)
        
        # Áp dụng Otsu Thresholding để phân đoạn ảnh
        otsu_mask = segment_by_otsu(gray_image)
        
        # Bước 2: Phân đoạn ảnh với KMeans
        segmented_image, kmeans_mask = segment_by_kmeans(image, k)
        
        # Bước 3: Kết hợp kết quả Otsu và KMeans
        # Giữ lại vùng có giá trị lớn hơn ngưỡng Otsu và những vùng phân đoạn qua KMeans
        combined_mask = otsu_mask & kmeans_mask  # Chỉ giữ lại các vùng mà cả Otsu và KMeans đều xác nhận
        
        # Áp dụng mask để giữ lại các vùng quan trọng
        masked_image = image.copy()
        masked_image[combined_mask == 0] = 0  # Giữ lại những vùng không phải là nền
        
        return masked_image, otsu_mask, kmeans_mask, combined_mask

    def preprocess_fundus_image(image):
        # Làm nét
        sharpened = apply_unsharp_mask(image)
        
        denoised = cv2.fastNlMeansDenoisingColored(sharpened, None, 10, 10, 7, 21)
        
        # Cải thiện độ tương phản bằng CLAHE
        clahe_img = apply_clahe(denoised)

        return clahe_img

    def contrast_feature(matrix):
        return np.mean(graycoprops(matrix, 'contrast'))

    def dissimilarity_feature(matrix):
        return np.mean(graycoprops(matrix, 'dissimilarity'))

    def homogeneity_feature(matrix):
        return np.mean(graycoprops(matrix, 'homogeneity'))

    def energy_feature(matrix):
        return np.mean(graycoprops(matrix, 'energy'))

    def correlation_feature(matrix):
        return np.mean(graycoprops(matrix, 'correlation'))

    def asm_feature(matrix):
        return np.mean(graycoprops(matrix, 'ASM'))
        
    def mean_feature(matrix):
        return np.mean(matrix)

    def variance_feature(matrix):
        return np.var(matrix)

    def sd_feature(matrix):
        return np.std(matrix)
        
    def rms_feature(matrix):
        return np.sqrt(np.mean(np.square(matrix)))
    
    def preprocessing_image():
        list_of_images = []
        path_of_images = os.path.join(base_path,file_path)
        images = os.listdir(path_of_images)
        list_of_images.extend([os.path.join(path_of_images, img) for img in images])

        total_time_preprocessing_image = 0
        total_time_segment_image = 0
        
        total_time_feature_selected_time = 0
        total_time_feature_extraction_time = 0

        feature_selected_time = time.time()
        if os.path.exists(os.path.join(base_path,f'data/Dataset_diabetic/{folder_save}/images_ft.csv')):
            os.remove(os.path.join(base_path,f'data/Dataset_diabetic/{folder_save}/images_ft.csv'))
        if not os.path.exists(os.path.join(base_path,f'data/Dataset_diabetic/{folder_save}/images_ft.csv')):
            with open(os.path.join(base_path,f'data/Dataset_diabetic/{folder_save}/images_ft.csv'), mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "image_id",
                    "Contrast Feature",
                    "Dissimilarity Feature",
                    "Homogeneity Feature",
                    "Energy Feature",
                    "Correlation Feature",
                    "ASM Feature",
                    "Mean Feature",
                    "Variance Feature",
                    "Standard Deviation Feature",
                    "RMS Feature"
                ])
        rs_feature_selected_time = time.time()
        total_time_feature_selected_time += rs_feature_selected_time-feature_selected_time

        for image in list_of_images:
            time_preprocess_image = time.time()
            img = cv2.imread(image)
            img = cv2.resize(img, (256, 256))
            img = preprocess_fundus_image(img)
            segmented_img, otsu_mask, kmeans_mask, combined_mask = segment_with_otsu_and_kmeans(img, k=3)
            total_time_preprocessing_image += time.time()-time_preprocess_image
            
            time_segment_image = time.time()
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Bước 5: Áp dụng mask Otsu để giữ lại vùng tổn thương
            masked_img = cv2.bitwise_and(gray_img, gray_img, mask=otsu_mask)

            # Bước 6: Chuyển ảnh masked về dạng 8-bit nếu cần
            masked_img_ubyte = img_as_ubyte(masked_img)

            # Bước 7: Phân bin ảnh thành 16 mức xám

            # Bước 8: Tính GLCM
            matrix_coocurrence = graycomatrix(
                masked_img_ubyte,
                distances=[1],
                angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                levels=256,
                normed=True,
                symmetric=True,
            )
            masked_inds = img[otsu_mask == 0]
            CF = contrast_feature(matrix_coocurrence)
            DF= dissimilarity_feature(matrix_coocurrence)
            HF = homogeneity_feature(matrix_coocurrence)
            EF = energy_feature(matrix_coocurrence)
            COR = correlation_feature(matrix_coocurrence)
            ASM = asm_feature(matrix_coocurrence)
            MF = mean_feature(masked_inds)
            VF = variance_feature(masked_inds)
            SD = sd_feature(masked_inds)
            RMS = rms_feature(masked_inds)
            
            feature_extraction_time = time.time()

            with open(os.path.join(base_path,f'data/Dataset_diabetic/{folder_save}/images_ft.csv'), mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    os.path.splitext(os.path.basename(image))[0],
                    CF,
                    DF,
                    HF,
                    EF,
                    COR,
                    ASM,
                    MF,
                    VF,
                    SD,
                    RMS
                ])
            rs_feature_extraction_time = time.time()
            total_time_feature_extraction_time += rs_feature_extraction_time-feature_extraction_time
            
            total_time_segment_image += time.time() - time_segment_image
            del img, gray_img, masked_img, masked_img_ubyte, matrix_coocurrence,CF,DF,HF,EF,COR,ASM,MF,VF,SD,RMS
            gc.collect()

            print(f'Total time processing image: {total_time_preprocessing_image}')
            print(f'Total time segment image: {total_time_segment_image}')
            
            print(f'Feature extraction image total time: {total_time_feature_extraction_time}')


            print(f'Feature selected image total time: {total_time_feature_selected_time}')

            time_image = total_time_preprocessing_image + total_time_segment_image + total_time_feature_extraction_time + total_time_feature_selected_time
            print(f'Time processing image: {time_image}')
            df = pd.read_csv(os.path.join(base_path,f'data/Dataset_diabetic/{folder_save}/images_ft.csv'))
            dfData = pd.read_csv(os.path.join(base_path,"data/Dataset_diabetic/labels_brset.csv"))
            columns_to_keep = [
                'image_id',
                'diabetic_retinopathy'
            ]
            dfData = dfData[columns_to_keep]
            dfMerge = dfData.merge(df,how = 'inner', on= 'image_id')
            dfMerge = dfMerge[[col for col in dfMerge.columns if col != 'diabetic_retinopathy'] + ['diabetic_retinopathy']]
            dfMerge.to_csv(os.path.join(base_path,f'data/Dataset_diabetic/{folder_save}/images_ft.csv'),index=False)
    def preprocessing_image_feature_selection():
        df = pd.read_csv(os.path.join(base_path,f'data/Dataset_diabetic/{folder_save}/images_ft.csv'))
        for col in df.columns[1:]:
            df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        index = df.iloc[:, 0]
        y = df.iloc[:, -1].values
        Fimg = df.iloc[:, 1:-1].values
        Fimg = pd.DataFrame(Fimg)
        Fimg.insert(0, 'image_id', index.values)
        Fimg = pd.concat([Fimg,pd.DataFrame(y)],axis=1)
        Fimg.to_csv(os.path.join(base_path,f'data/Dataset_diabetic/{folder_save}/image_fts_norm.csv'),index=False)
    # preprocessing_image()
    preprocessing_image_feature_selection()
def processing_wrapper(file_path_img,file_path_table, folder_save,max_img,max_tab):
    os.makedirs(os.path.join(base_path,f'data/Dataset_diabetic/{folder_save}'), exist_ok=True)
    process_image(file_path=file_path_img,folder_save=folder_save)
    df_image = pd.read_csv(f'data/Dataset_diabetic/{folder_save}/image_fts_norm.csv')
    # df_merge = df_image.merge(df_table.iloc[:,:-1],how='inner',on="image_id")
    # df_merge = df_merge.drop(['image_id'],axis=1)
    
    Label = df_image.iloc[:,-1]
    mid = df_image.shape[1] // 2

    Ftab = df_image.iloc[:, 1:mid]
    Fimg = df_image.iloc[:, mid:-1]
    
    from sklearn.preprocessing import StandardScaler
    scaler_img = StandardScaler()
    scaler_tab = StandardScaler()
    Fimg = scaler_img.fit_transform(Fimg.values)
    Ftab = scaler_tab.fit_transform(Ftab.values)
    
    # Áp dụng wrapper
    from module.Processing_Data.Fusion_function import wrapper_multimodal_feature_selection
    Ffused_df = pd.DataFrame(wrapper_multimodal_feature_selection.wrapper_multimodal_selection(Fimg=Fimg,Ftab=Ftab,target=Label.values.ravel(),max_img=max_img,max_tab=max_tab,min_img=2,min_tab=2))
    Ffused_df = pd.concat([Ffused_df, Label.reset_index(drop=True)], axis=1)
    
    X = Ffused_df.iloc[:, :-1]
    y = Ffused_df.iloc[:, -1]
    X.columns = X.columns.astype(str)
    # Áp dụng BorderlineSMOTE
    # Xử lý dữ liệu mất cân bằng
    border_smote = BorderlineSMOTE(random_state=42, sampling_strategy='auto', k_neighbors=5)
    X_resampled, y_resampled = border_smote.fit_resample(X, y)

    # Gộp lại thành DataFrame mới
    dfBalanced = pd.concat([
        pd.DataFrame(X_resampled, columns=X.columns),
        pd.Series(y_resampled, name='diabetic_retinopathy')
    ], axis=1)
    dfBalanced.to_csv(os.path.join(base_path,f'data/Dataset_diabetic/{folder_save}/data_process.csv'),index=False)
def processing(file_path_img,file_path_table, folder_save):
    os.makedirs(os.path.join(base_path,f'data/Dataset_diabetic/{folder_save}'), exist_ok=True)
    process_image(file_path=file_path_img,folder_save=folder_save)
    df_image = pd.read_csv(f'data/Dataset_diabetic/{folder_save}/image_fts_norm.csv')
    
    X = df_image.iloc[:, 1:-1]
    y = df_image.iloc[:, -1]
    X.columns = X.columns.astype(str)
    # Áp dụng BorderlineSMOTE
    # Xử lý dữ liệu mất cân bằng
    border_smote = BorderlineSMOTE(random_state=42, sampling_strategy='auto', k_neighbors=5)
    X_resampled, y_resampled = border_smote.fit_resample(X, y)

    # Gộp lại thành DataFrame mới
    dfBalanced = pd.concat([
        pd.DataFrame(X_resampled, columns=X.columns),
        pd.Series(y_resampled, name='diabetic_retinopathy')
    ], axis=1)
    dfBalanced.to_csv(os.path.join(base_path,f'data/Dataset_diabetic/{folder_save}/data_process.csv'),index=False)
    