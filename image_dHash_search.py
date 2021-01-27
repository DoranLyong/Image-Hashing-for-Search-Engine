# -*- coding: utf-8 -*- 
"""
(ref) https://www.pyimagesearch.com/2017/11/27/image-hashing-opencv-python/

(ref) 이미지 Hash란 무엇인가?: https://www.kaggle.com/namepen/image-hash
(ref) intro to Hashing: https://www.2brightsparks.com/resources/articles/introduction-to-hashing-and-its-uses.html
(ref) List of hash functions: https://en.wikipedia.org/wiki/List_of_hash_functions
(ref) dHash: http://www.hackerfactor.com/blog/index.php?/archives/529-Kind-of-Like-That.html
"""



# %% 임포트 패키지 
import sys 
import os.path as osp 
from pathlib import Path 
import time

from tqdm import tqdm
from glob import glob 
import cv2 
import matplotlib.pyplot as plt



# ================================================================= #
#                    1. Define dhash algorithm                      #
# ================================================================= #
# %% 01. dHash 알고리즘 정의 

def dHash(image, hashSize=8): 
    """
    - the image is gray-level
    - hashSize = 8   =>  the output hash will be 8 x 8 = 64-bit


    1. resize the input image
    2. add a single column (width) so we can compute the horizontal gradient
    """
    resized = cv2.resize(image, (hashSize + 1, hashSize))
    

    """
    3. compute the (relative) horizontal gradient between adjacent column pixels
    """
    diff = resized[:, 1:] > resized[:, :-1]


    """
    4. convert the difference image to a hash
    """
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])



# ================================================================= #
#                           2. DataLoader                           #
# ================================================================= #
# %% 02. 데이터 업로드 
"""
Query 와 Gallery 이미지의 경로 리스트를 구한다 
"""

#DB_root = osp.join("dataset", "DukeMTMC-reID")
DB_root = osp.join("image-hashing-opencv")

galleryPath =  osp.join(DB_root, "gallery")
queryPath = osp.join(DB_root, "query")

gallery_folders = sorted(glob(osp.join(galleryPath, "*")))
query_folders = sorted(glob(osp.join(queryPath, "*")))



gallery_imgs = [img_path for gf in gallery_folders  for img_path in glob(osp.join(gf, "*.jpg"))]
query_imgs = [img_path for qf in query_folders for img_path in glob(osp.join(qf, "*.jpg"))]



# ================================================================= #
#                          3. Compute dHash                           #
# ================================================================= #
# %% image hash code 얻기 

# grab the paths to both the gallery and query images 
print("[INFO] computing hashes for the gallery...")


gallery_HashStack = {}

start = time.time() 

for p in tqdm(gallery_imgs):
    img = cv2.imread(p)

    if img is None:
        # if the image is None then we could not load it from disk (so skip it)
        continue


    """dHash process 
    """ 
    gray_level = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgHash = dHash(gray_level, hashSize=8)


    """ update the gallery dictionary 
    """
    l = gallery_HashStack.get(imgHash, [])
    l.append(p) 
    gallery_HashStack[imgHash] = l



print(f"[INFO] processed {len(gallery_HashStack)} images in {time.time() - start:.2f} seconds")


print("[INFO] computing hashes for the querys...")

for p in query_imgs:
    img = cv2.imread(p)

    if img is None:
        # if the image is None then we could not load it from disk (so skip it)
        continue

    """dHash process 
    """ 
    gray_level = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgHash = dHash(gray_level, hashSize=8)



    """ grab all image paths that match the hash.
    Query 와 동일한 이미지를 Gallery 에서 가져오기 
    """
    matchedPaths = gallery_HashStack.get(imgHash, [])


    for target in matchedPaths:
        b = p.split(osp.sep)[-2]
        print(f"Query: {b}")


        print(f"Find here: {target}")


# %%
"""dHash
하지만, ReID DB를 대상으로 dHash 알고리즘을 적용했을 때 성능이 안 나옴.

왜냐하면 완벽히 동일한 이미지가 아니고서는 dHash code가 서로 다르고,
서로 전부 고유한 코드번호를 가지고 있고 때문. 
"""
