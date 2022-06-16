# image-processing
**   example1**
import cv2
img=cv2.imread('plants.jpg',0)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
**  output**
![image](https://user-images.githubusercontent.com/104187589/173810494-0f2e7467-802a-4dbd-ae8f-b14342ab748b.png)


**  example 2**
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
img=mpimg.imread('leaf.jpg')
plt.imshow(img)
**  output**
![image](https://user-images.githubusercontent.com/104187589/173809911-dd6192ba-ac88-4f1d-9fa4-05e572091e54.png)


**  example3**
from PIL import Image
img=Image.open("butterfly.jpg")
img=img.rotate(180)
img.show()
cv2.waitKey(0)
cv2.destroyAllwindows()
**  output**
![image](https://user-images.githubusercontent.com/104187589/173813639-77c63461-9337-4783-a3ab-60d7045fb357.png)


** example4**
from PIL import ImageColor
# using getrgb for yellow
img1=ImageColor.getrgb("yellow")
print(img1)
#using getrgb for red
img2=ImageColor.getrgb("red")
print(img2)

**  output**
(255, 255, 0)
(255, 0, 0)

**  example5**
from PIL import Image 
img=Image.new('RGB',(200,400),(255,255,0))
img.show()
**  output**
![image](https://user-images.githubusercontent.com/104187589/173817391-7d4a1af7-ef11-4648-a84a-0c1bf0eb0e40.png)

**  example6**
import cv2
import matplotlib.pyplot as plt
import numpy as np
img=cv2.imread('butterfly.jpg')
plt.imshow(img)
plt.show()
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()
img=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
plt.imshow(img)
plt.show()
img=cv2.cvtColor(img,cv2.COLOR_CMYK )
plt.imshow(img)
plt.show()

**  output**
![image](https://user-images.githubusercontent.com/104187589/174037981-b10e816e-a6f0-4bed-bd65-081c3f8a2473.png)
![image](https://user-images.githubusercontent.com/104187589/174038161-41ef87d0-595a-447d-b840-78b98ec7b0d6.png)
![image](https://user-images.githubusercontent.com/104187589/174038393-a6fb31ff-8041-4713-9c2b-abcf07557ecd.png)

**  example 7**
from PIL import Image
image=Image.open('flower.jpg')
print("Filename:",image.filename)
print("Format:",image.format)
print("Mode:",image.mode)
print("Size:",image.size)
print("Width:",image.width)
print("Height:",image.height)
image.close()

**   output **
Filename: flower.jpg
Format: JPEG
Mode: RGB
Size: (474, 474)
Width: 474
Height: 474

** example 8**
import cv2
img=cv2.imread('flower.jpg')
print('original image length width',img.shape)
cv2.imshow('original image',img)
#to show the resized image
imgresize=cv2.resize(img,(150,160))
cv2.imshow('Resized image',imgresize)
print('Resized image length width',imgresize.shape)
cv2.waitKey(0)

** output**
![image](https://user-images.githubusercontent.com/104187589/174047283-fc01856a-6116-4b59-945a-c62b10eb560f.png)
![image](https://user-images.githubusercontent.com/104187589/174047408-0972ed84-d246-476a-b7c9-b9e9da0a2279.png)
![image](https://user-images.githubusercontent.com/104187589/174047576-971e02bf-902d-4b28-85a1-f7cd23865bfd.png)





