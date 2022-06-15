# image-processing
**   example1**
import cv2
img=cv2.imread('flower.jpg',0)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
**  output**


**  example 2**
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
img=mpimg.imread('leaf.jpg')
plt.imshow(img)
**  output**
![image](https://user-images.githubusercontent.com/104187589/173809911-dd6192ba-ac88-4f1d-9fa4-05e572091e54.png)
