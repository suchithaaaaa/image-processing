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

**  example 9**
import cv2
# read the image file
img=cv2.imread('flower.jpg')
cv2.imshow("RGB",img)
cv2.waitKey(0)
# Gray scale
img=cv2.imread('flower.jpg',0)
cv2.imshow("Gray",img)
cv2.waitKey(0)
# Binary image
ret,bw_img=cv2.threshold(img,127,255,cv2.THRESH_BINARY)
cv2.imshow("Binary",bw_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

**  output**
![image](https://user-images.githubusercontent.com/104187589/174054528-f0daa172-ec3c-462e-bfe3-730f74bbab44.png)
![image](https://user-images.githubusercontent.com/104187589/174054693-2dc0061f-289a-4c9b-9bda-3273db11b105.png)
![image](https://user-images.githubusercontent.com/104187589/174056336-c10b27d1-bd07-4371-b189-328744b0f156.png)

** 10 Develop a program to readimage using URL**
from skimage import io
import matplotlib.pyplot as plt
url='https://www.thesprucepets.com/thmb/FOLwbR72UrUpF9sZ45RYKzgO8dg=/3072x2034/filters:fill(auto,1)/yellow-tang-fish-508304367-5c3d2790c9e77c000117dcf2.jpg'
image=io.imread(url)
plt.imshow(image)
plt.show()

**  output **
![image](https://user-images.githubusercontent.com/104187589/175007465-0d9b43b2-313e-4440-8b70-22a3e07eea10.png)

** 11 mask and blur the image**
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
img=cv2.imread('fish2.jpg')
plt.imshow(img)
plt.show()

**  output**
![image](https://user-images.githubusercontent.com/104187589/175264380-a5c6dff9-d824-4efc-bcd1-9073007e11db.png)

import cv2
hsv_img=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
light_orange=(1,190,200)
dark_orange=(418,255,255)
mask=cv2.inRange(img,light_orange,dark_orange)
result=cv2.bitwise_and(img,img,mask=mask)
plt.subplot(1,2,1)
plt.imshow(mask,cmap="gray")
plt.subplot(1,2,2)
plt.imshow(result)
plt.show()

**  output**
![image](https://user-images.githubusercontent.com/104187589/175264695-9a14b7e5-7991-4678-92e8-05fecb775404.png)


light_white=(0,0,200)
dark_white=(145,60,255)
mask_white=cv2.inRange(hsv_img,light_white,dark_white)
result_white=cv2.bitwise_and(img,img,mask=mask_white)
plt.subplot(1,2,1)
plt.imshow(mask_white,cmap="gray")
plt.subplot(1,2,2)
plt.imshow(result_white)
plt.show()

** output**
![image](https://user-images.githubusercontent.com/104187589/175264932-5fd19f6a-76ca-4e77-b96f-ac0f0fc02016.png)

final_mask=mask+mask_white
final_result=cv2.bitwise_and(img,img,mask=final_mask)
plt.subplot(1,2,1)
plt.imshow(final_mask,cmap="gray")
plt.subplot(1,2,2)
plt.imshow(final_result)
plt.show()

**  output**
![image](https://user-images.githubusercontent.com/104187589/175265288-7dda1385-4979-426a-a37b-f5d71df2f365.png)

blur=cv2.GaussianBlur(final_result,(7,7),0)
plt.imshow(blur)
plt.show()

**  output**
![image](https://user-images.githubusercontent.com/104187589/175265601-8021868f-10d7-4d3c-995a-44bc183076e3.png)


** 12  perform arithmetic operation on images**
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
#Reading image files
img1=cv2.imread('img.jpg')
img2=cv2.imread('imgr.jpg')
# Applying NumPy addition on images
fimg1=img1+img2
plt.imshow(fimg1)
plt.show()
#saving the output image
cv2.imwrite('output.jpg',fimg1)
fimg2=img1-img2
plt.imshow(fimg2)
plt.show()
#saving the output image
cv2.imwrite('output.jpg',fimg2)
fimg3=img1*img2
plt.imshow(fimg3)
plt.show()
#saving the output image
cv2.imwrite('output.jpg',fimg3)
fimg4=img1/img2
plt.imshow(fimg4)
plt.show()
#saving the output image
cv2.imwrite('output.jpg',fimg4)

**  output**
![image](https://user-images.githubusercontent.com/104187589/175271927-7c7cc9bd-34f5-43fe-bc3c-e99a6c4ff7d2.png)

![image](https://user-images.githubusercontent.com/104187589/175272144-3a1e86fd-0a76-42fb-a3fb-5ba83c12bb91.png)

![image](https://user-images.githubusercontent.com/104187589/175272238-dd2f523b-be45-424e-ba9f-7dd77143bc34.png)

** 13 change the images to different color**
import cv2 as c
import numpy as np
from PIL import Image
array = np.zeros( [100,200,3],dtype=np.uint8)
array[:,:100]=[255,130,0]
array[:,100:]=[0,0,255]
img=Image.fromarray(array)
img.save('image1.png')
img.show()
c.waitKey(0)

**  output**
![image](https://user-images.githubusercontent.com/104187589/175282335-b60e4cfe-8e5c-4ae0-ae0b-f5da7b80dc37.png)


** 14 create an image using 2D array**
import cv2
img=cv2.imread('E://butterfly.jpg')
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
lab=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
hls=cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
yuv=cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
cv2.imshow("GRAY image",gray)
cv2.imshow("HSV image",hsv)
cv2.imshow("LAB image",lab)
cv2.imshow("HLS image",hls)
cv2.imshow("YUV image",yuv)
cv2.waitKey(0)
cv2.destroyAllWindows()


**  output**
![image](https://user-images.githubusercontent.com/104187589/175287197-2660046c-c4d3-4dff-96d2-18c724740159.png)
![image](https://user-images.githubusercontent.com/104187589/175287269-23f1ae5b-51e7-486e-82bb-b0113c2cb8bf.png)
![image](https://user-images.githubusercontent.com/104187589/175287371-2a457d95-f19c-47a2-84be-8995c03893fb.png)
![image](https://user-images.githubusercontent.com/104187589/175287457-7f7bcfb0-def7-4504-93a3-66ba43aa0469.png)
![image](https://user-images.githubusercontent.com/104187589/175287557-469a3c5a-a47f-456f-acbf-495b3c4a8adb.png)

** 15 bitwise operation **
import cv2
import matplotlib.pyplot as plt
image1=cv2.imread('plants.jpg',1)
image2=cv2.imread('plants.jpg')
ax=plt.subplots(figsize=(15,10))
bitwiseAnd=cv2.bitwise_and(image1,image2)
bitwiseOr=cv2.bitwise_or(image1,image2)
bitwiseXor=cv2.bitwise_xor(image1,image2)
bitwiseNot_img1=cv2.bitwise_not(image1)
bitwiseNot_img2=cv2.bitwise_not(image2)
plt.subplot(151)
plt.imshow(bitwiseAnd)
plt.subplot(152)
plt.imshow(bitwiseOr)
plt.subplot(153)
plt.imshow(bitwiseXor)
plt.subplot(154)
plt.imshow(bitwiseNot_img1)
plt.subplot(155)
plt.imshow(bitwiseNot_img2)
cv2.waitKey(0)

**  output**
![image](https://user-images.githubusercontent.com/104187589/176410181-a04796fb-f8da-47ca-abe1-84f74c291f91.png)

** 16 blur image**
# importing  libraries
import cv2
import numpy as np
image=cv2.imread('flowers.jpg')
cv2.imshow('original image',image)
cv2.waitKey(0)
# Gaussian Blur
Gaussian=cv2.GaussianBlur(image,(7,7),0)
cv2.imshow('Gaussian Blurring',Gaussian)
cv2.waitKey(0)
# Median Blur
median=cv2.medianBlur(image,5)
cv2.imshow('Median Blurring',median)
cv2.waitKey(0)
#Bilateral Blur
bilateral=cv2.bilateralFilter(image,9,75,75)
cv2.imshow('Bilateral Blurring',bilateral)
cv2.waitKey(0)
cv2.destroyAllWindows()

** output**
![image](https://user-images.githubusercontent.com/104187589/176414183-12d9a17f-c335-483f-88d5-eeba261be867.png)
![image](https://user-images.githubusercontent.com/104187589/176414282-d5b3f541-dd94-4ef5-9bd3-5d809bee64a0.png)
![image](https://user-images.githubusercontent.com/104187589/176414380-e63b2794-ce7b-41e7-bd00-c34323154853.png)
![image](https://user-images.githubusercontent.com/104187589/176414541-5bfe9741-38ae-45d6-81f9-a06437e90895.png)





















