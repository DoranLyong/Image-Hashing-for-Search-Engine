# -*- coding: utf-8 -*- 
"""
(ref) https://www.pyimagesearch.com/2017/11/27/image-hashing-opencv-python/

(ref) 이미지 Hash란 무엇인가?: https://www.kaggle.com/namepen/image-hash
(ref) intro to Hashing: https://www.2brightsparks.com/resources/articles/introduction-to-hashing-and-its-uses.html
(ref) List of hash functions: https://en.wikipedia.org/wiki/List_of_hash_functions
(ref) dHash: http://www.hackerfactor.com/blog/index.php?/archives/529-Kind-of-Like-That.html
"""





#%% 임포트 패키지 
import sys 
import os.path as osp 
import argparse

import cv2 
import matplotlib.pyplot as plt




# ================================================================= #
#                    1. Define dhash algorithm                      #
# ================================================================= #
# %% 01. dHash 알고리즘 정의 

def dhash(image, hashSize=8): 

    plt.imshow(image)
    plt.axis("off")
    plt.show()
    plt.close()

    """
    - the image is gray-level
    - hashSize = 8   =>  the output hash will be 8 x 8 = 64-bit


    1. resize the input image
    2. add a single column (width) so we can compute the horizontal gradient
    """
    resized = cv2.resize(image, (hashSize + 1, hashSize))
    
    plt.imshow(resized)
    plt.axis("off")
    plt.show()
    plt.close()
    print(resized.shape)


    """
    3. compute the (relative) horizontal gradient between adjacent column pixels
    """
    diff = resized[:, 1:] > resized[:, :-1]


    plt.imshow(diff)
    plt.axis("off")
    plt.show()
    plt.close()
    print(diff.shape)

    """
    4. convert the difference image to a hash
    """
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])



# ================================================================= #
#                        2. Hyperparameters                         #
# ================================================================= #
# %% 02. 하이퍼파라미터 설정 
DB_root = osp.join("dataset")

galleryPath =  osp.join(DB_root, "gallery")
queryPath = osp.join(DB_root, "query")





# ================================================================= #
#                          3. Processing                            #
# ================================================================= #
# %% 03. 하이퍼파라미터 설정 
img = cv2.imread("0022_c4_f0031602.jpg")
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imageHash = dhash(gray_img, hashSize=10)


# %%
print(img.shape)
print(imageHash)
print(len(bin(imageHash)))
print(imageHash.bit_length())
# %%





