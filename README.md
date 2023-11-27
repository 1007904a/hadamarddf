# Design of a Brief Perceptual Loss Function with Hadamard Codes

## Abstract
Perceptual loss functions are central to an ever-increasing number of tasks across computer vision. Their strength lies in their ability to translate perceptual nuances into numerical high-level features. A cornerstone of these functions are the high-dimensional, real-valued deep feature vectors. However, their memory-intensive nature often hinders deployment on devices with constrained resources.

We introduce a concise perceptual loss function underpinned by Hadamard codes. For the ImageNet collection, our method delivers a lean representation of a mere 128 bytes. Impressively, this representation is not tied to any specific architecture, paving the way for the integration of industry-standard models.

Utilizing our proposed binary codes in conjunction with 
$k$NN and Half-Space Proximal (HSP) classifiers (with HSP being a noteworthy alternative to 
$k$NN), we have secured commendable accuracy. This novel approach sets new benchmarks, enhancing state-of-the-art performance in knowledge transfer across a variety of image datasets.

## Libraries
[NMSLIB](https://github.com/nmslib/nmslib.git)  
[faiss from FaceBook](https://github.com/facebookresearch/faiss.git)  
h5py==3.9.0    
pytorch==2.0.1    
pytorch-cuda==11.8     
scikit-learn==1.2.2    
scipy==1.10.1   
sklearn==0.0.post5
torchaudio==2.0.2    
torchtriton==2.0.0    
torchvision==0.15.2   

## Files description

(1) train_dhf_ver01.py and train_dhf_ver02.py.  These files re-train a CNN model with Hadamard Codes, by default we left resnet101 model configuration ready to be re-trained.

The cnn model re-trained it's saved in 'checkpoint/{name_file}'.

(2) extract_df.py, extract_dhf.py, extract_df_quantized.py and extract_df_w_matrix.py. These files extract Deep Features (df) or Deep Hadamard Features (df) from the last layer before output's model.
The models used in these files are:

| Index | Model Name      | Code           |
|:------|:----------------|:---------------|
| 0     | EfficientNet B0 | ef0            |
| 1     | EfficientNet B1 | ef1            |
| 2     | EfficientNet B2 | ef2            |
| 3     | EfficientNet B3 | ef3            |
| 4     | ResNet 50       | resnet50       |
| 5     | Resnet 101      | resnet101      |
| 6     | VGG 16          | vgg16          |
| 7     | Mnasnet1 3      | mnasnet1_3     |
| 8     | Convnext Large  | convnext_large |
| 9     | Vit h 14        | vit_h_14       |
|10     | Regnet y 128gf  | regnet_y_128gf |
|11     | Maxvit t        | maxvit_t       |
|12     | Swin v2 b       |	swin_v2_b      |

In this step, "corr_mtrx_00_{Code}/[txt_classes.txt | txt_vectors.txt]" files are created.

(3) df_exhaustive_search.py and df_exhaustive_search_faiss.py. In order to get nearest neighborns from a query $q$ with DF and DHF, we used NMSLIB and Faiss indexers to reduce computer time. 
The code in df_exhaustive_search.py and df_exhaustive_search_faiss.py show NMSLIB and Faiss implementation with default parameters.

In this step, a file with $k$ nearest neighborns for each imagen in evaluation set (every imagen as query $q$) is created.
This file is saved in "index/{file_name}.txt".

(4) binary_exhaustive_search.cpp and binary_exhaustive_search.h. The binary_exhaustive_search.cpp and binary_exhaustive_search.h files are an implementation to get nearest neighborns with Deep Hadamard Features. 
In our implementation, we used bits operations with "builtin_popcountll" function in C++.

To compile and run:
```bash
g++ -O3 -o binary_exhaustive_search binary_exhaustive_search.cpp && ./binary_exhaustive_search
```

(5) get_knn_recall.py and knn_library.py. To evaluate our work, we calculate recall @1, @5 and @10 from $k$-NN and [HSP classifier](https://link.springer.com/chapter/10.1007/11795490_19)


(6) create_h5_files.py. This file create a h5 file from data extracted from extract_df.py, extract_dhf.py, extract_df_quantized.py and extract_df_w_matrix.py files.
This data is saved in the 'labels' and 'data' tags from h5 file.

## Contact

Bryan Quiroz, CICESE Research Center, Mexico, quirozb@cicese.edu.mx

Bryan Martinez, FIE Universidad Michoacana, Mexico, emartinez@dep.fie.umich.mx

Antonio Camarena-Ibarrola, FIE Universidad Michoacana, Mexico, antonio.camarena@umich.mx

Edgar Chavez, CICESE Research Center, Mexico, elchavez@cicese.edu.mx
