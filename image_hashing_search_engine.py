# -*- coding: utf-8 -*- 
"""
(ref) https://www.pyimagesearch.com/2019/08/26/building-an-image-hashing-search-engine-with-vp-trees-and-opencv/

(ref) 이미지 Hash란 무엇인가?: https://www.kaggle.com/namepen/image-hash
(ref) intro to Hashing: https://www.2brightsparks.com/resources/articles/introduction-to-hashing-and-its-uses.html
(ref) List of hash functions: https://en.wikipedia.org/wiki/List_of_hash_functions
(ref) dHash: http://www.hackerfactor.com/blog/index.php?/archives/529-Kind-of-Like-That.html


Image hashing algorithms are used to:

(1) Uniquely quantify the contents of an image using only a single integer.

(2) Find duplicate or near-duplicate images in a dataset of images based on their computed hashes.



[Linear search  -> VP-Tree ]

Using a VP-Tree we can reduce our search complexity from O(n) to O(log n), 
enabling us to obtain our sub-linear goal!


[In this tutorial]:
(1) Build an image hashing search engine to find both identical and near-identical images in a dataset.

(2) Utilize a specialized data structure, called a VP-Tree, 
    that can be used used to scale image hashing search engines to millions of images.

"""



# %% 임포트 패키지 
import sys 
import os.path as osp 
import time

from tqdm import tqdm
from glob import glob 
import cv2 
import matplotlib.pyplot as plt
import numpy as np 
import vptree
import pickle



# ================================================================= #
#                           Hyperparameters                         #
# ================================================================= #
hashSize = 128



# ================================================================= #
#                 1. Define dHash & Hamming distance                #
# ================================================================= #
# %% 01. dHash 알고리즘 & Hamming distance 정의 

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


def convert_hash(h):
	# convert the hash to NumPy's 64-bit float and then back to
	# Python's built in int
	return int(np.array(h, dtype="float64"))


def hamming(a, b):
	# compute and return the Hamming distance between the integers
	return bin(int(a) ^ int(b)).count("1")




# ================================================================= #
#                           2. DataLoader                           #
# ================================================================= #
# %% 02. 데이터 업로드 
"""
Query 와 Gallery 이미지의 경로 리스트를 구한다 
"""

DB_root = osp.join("dataset", "DukeMTMC-reID")
#DB_root = osp.join("image-hashing-opencv")

galleryPath =  osp.join(DB_root, "gallery")
queryPath = osp.join(DB_root, "query")

gallery_folders = sorted(glob(osp.join(galleryPath, "*")))
query_folders = sorted(glob(osp.join(queryPath, "*")))



gallery_imgs = [img_path for gf in gallery_folders  for img_path in glob(osp.join(gf, "*.jpg"))]
query_imgs = [img_path for qf in query_folders for img_path in glob(osp.join(qf, "*.jpg"))]




# ================================================================= #
#                3. Building Hash DB using VP-Tree                  #
# ================================================================= #
# %% 03. VP-Tree 구조로 Hash 데이터베이스 만들기 


gallery_hashes = {} 

for i, imgPath in enumerate(gallery_imgs):
    print(f"[INFO] processing image {i + 1}/{len(gallery_imgs)}")

    img = cv2.imread(imgPath)

    if img is None: 
        # if the image is None then we could not load it from disk (so skip it)
        continue

    
    """dHash process 
    """     
    h = dHash(img, hashSize=hashSize)
#    h = convert_hash(h)


    """ update the hashes dictionary 
    """
    l = gallery_hashes.get(h, [])
    l.append(imgPath)
    gallery_hashes[h] = l


""" Build the VP-Tree
"""
print("[INFO] building VP-Tree...")
points = list(gallery_hashes.keys())
tree = vptree.VPTree(points, hamming)



""" serialize the VP-Tree to disk
"""
print("[INFO] serializing VP-Tree...")
f = open("vptree.pickle", "wb")
f.write(pickle.dumps(tree))
f.close()


""" serialize the hashes to dictionary
"""
print("[INFO] serializing hashes...")
f = open("gallery_hashes.pickle", "wb")
f.write(pickle.dumps(gallery_hashes))
f.close()



# ================================================================= #
#         4. Implementing the image hash searching engine           #
# ================================================================= #
# %% 04. Searching Engine 만들기 

print("[INFO] loading VP-Tree and hashes...")

tree = pickle.loads(open("vptree.pickle", "rb").read())
DB_hashes = pickle.loads(open("gallery_hashes.pickle", "rb").read())



""" load the input query image
"""
query_sample = query_imgs[30]
image = cv2.imread(query_sample)

cv2.imwrite("Query.jpg", image)



""" compute the hash for the query image, then convert it
"""
queryHash = dHash(image, hashSize=hashSize)
#queryHash = convert_hash(queryHash)




""" perform the search
"""
print("[INFO] performing search...")
start = time.time()

distance = 21700
results = tree.get_all_in_range(queryHash, distance)
results = sorted(results)

end = time.time()
print("[INFO] search took {} seconds".format(end - start))



for dst, hash in results:
    """ grab all image paths in our dataset with the same hash
    """
    resultPaths = DB_hashes.get(hash, [])
#    print(f"[INFO] {len(resultPaths)} total image(s) with dst: {dst}, hash: {hash}")
    print(f"[INFO] {len(resultPaths)} total image(s) with dst: {dst}")


    """ loop over the result paths
    """
    for resultPath in resultPaths:
        print(f"[INFO] Retrieve from {resultPath}")
        result = cv2.imread(resultPath)

        cv2.imwrite(f"Retrieve_dst_{dst}.jpg", result)


# %%

"""
아무튼 ReID DB 를 적용하면,
잘 안 됨. 
"""
