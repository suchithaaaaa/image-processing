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

