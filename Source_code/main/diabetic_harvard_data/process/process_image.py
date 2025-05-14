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

base_path = Path(__file__).resolve().parents[1]

def apply_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_img = clahe.apply(image)

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
    Z = image.reshape((-1, 1))
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
    # gray = color.rgb2gray(image)
    gray_image = img_as_ubyte(image)
    
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
    sharpened = np.uint8(np.clip(sharpened, 0, 255))  # Đảm bảo ảnh trong khoảng [0, 255]
    denoised = cv2.fastNlMeansDenoising(sharpened, None, 10, 7, 21)
    
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

def preprocessing_image(file_path,folder_save):
    list_of_images = []
    path_of_images = os.path.join(base_path,file_path)
    
    images = os.listdir(path_of_images)
    list_of_images.extend([
    os.path.join(path_of_images, img) 
    for img in images 
    if img.endswith('.npz')
    ])
    # if os.path.exists(os.path.join(base_path,f'data/{folder_save}/images_ft.csv')):
    #     os.remove(os.path.join(base_path,f'data/{folder_save}/images_ft.csv'))
    if not os.path.exists(os.path.join(base_path,f'data/{folder_save}/images_ft.csv')):
        with open(os.path.join(base_path,f'data/{folder_save}/images_ft.csv'), mode='w', newline='') as f:
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

    for image in list_of_images:
        data= np.load(image)
        slo_fundus_image = data['slo_fundus']
        img = cv2.resize(slo_fundus_image, (200, 200))
        img = preprocess_fundus_image(img)
        segmented_img, otsu_mask, kmeans_mask, combined_mask = segment_with_otsu_and_kmeans(img, k=3)
        
        # Bước 5: Áp dụng mask Otsu để giữ lại vùng tổn thương
        masked_img = cv2.bitwise_and(img, img, mask=otsu_mask)

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
        
        
        with open(os.path.join(base_path,f'data/{folder_save}/images_ft.csv'), mode='a', newline='') as f:
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

        del img, masked_img, masked_img_ubyte, matrix_coocurrence,CF,DF,HF,EF,COR,ASM,MF,VF,SD,RMS
        gc.collect()

        df = pd.read_csv(os.path.join(base_path,f'data/{folder_save}/images_ft.csv'))
        
def preprocessing_image_oct(file_path,folder_save):
    list_of_images = []
    path_of_images = os.path.join(base_path,file_path)
    
    images = os.listdir(path_of_images)
    list_of_images.extend([
    os.path.join(path_of_images, img) 
    for img in images 
    if img.endswith('.npz')
    ])
    # if os.path.exists(os.path.join(base_path,f'data/{folder_save}/images_ft.csv')):
    #     os.remove(os.path.join(base_path,f'data/{folder_save}/images_ft.csv'))
    if not os.path.exists(os.path.join(base_path,f'data/{folder_save}/images_ft.csv')):
        with open(os.path.join(base_path,f'data/{folder_save}/images_ft.csv'), mode='w', newline='') as f:
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

    for image in list_of_images:
        data= np.load(image)
        oct_bscans_image = data['oct_bscans']
        avg_img = np.mean(oct_bscans_image, axis=0)
        img = cv2.resize(avg_img, (200, 200))
        img = preprocess_fundus_image(img)
        segmented_img, otsu_mask, kmeans_mask, combined_mask = segment_with_otsu_and_kmeans(img, k=3)
        
        # Bước 5: Áp dụng mask Otsu để giữ lại vùng tổn thương
        masked_img = cv2.bitwise_and(img, img, mask=otsu_mask)

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
                
        with open(os.path.join(base_path,f'data/{folder_save}/images_ft.csv'), mode='a', newline='') as f:
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
        
        del img, masked_img, masked_img_ubyte, matrix_coocurrence,CF,DF,HF,EF,COR,ASM,MF,VF,SD,RMS
        gc.collect()

        df = pd.read_csv(os.path.join(base_path,f'data/{folder_save}/images_ft.csv'))
