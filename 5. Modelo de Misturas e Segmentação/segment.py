from mixture_ import GMM
import pandas as pd
import numpy as np
import cv2 as cv
'''
image = cv.imread("brasil.PNG")

gmm = GMM(2, 100)

result = gmm.fit(image.reshape((image.size, 1)))
threshold = np.mean(result[0])
binary_img = image > threshold
cv.imshow('isso', binary_img)

cv.waitKey(0)
'''
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

np.random.seed(1)
n = 10
l = 256
im = np.zeros((l, l))
points = l*np.random.random((2, n**2))
im[(points[0]).astype(np.int), (points[1]).astype(np.int)] = 1
img = cv.imread("brasil.PNG")

width = img.shape[0]
height = img.shape[1]

dataset = []

separeted = []
for w in range(width):
    for h in range(height):
        dataset.append(img[w,h,:])

dataset = np.array(dataset)
print(np.unique(dataset, axis = 0))


classif = GMM(3, 1000)
#img = cv.resize(img, (30,30))
result = classif.fit(dataset[:10000,:])
print("Medias:", result[0])
#threshold = np.mean(result[0])
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
print(np.mean(result[0][0]))
binary_img_1 = img < np.mean(result[0][0])
binary_img_2 = img > np.mean([0 ,222 ,251])


plt.figure(figsize=(11,4))

plt.subplot(131)
plt.imshow(img)
plt.axis('off')
plt.subplot(132)
plt.imshow(binary_img_1, cmap=plt.cm.gray, interpolation='nearest')
plt.axis('off')
plt.subplot(133)
plt.imshow(binary_img_2, cmap=plt.cm.gray, interpolation='nearest')
plt.axis('off')

plt.subplots_adjust(wspace=0.02, hspace=0.3, top=1, bottom=0.1, left=0, right=1)
plt.show()