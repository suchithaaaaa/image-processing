# image-processing
**   example1**<br>
import cv2<br>
img=cv2.imread('plants.jpg',0)<br>
cv2.imshow('image',img)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>
**  output**<br>
![image](https://user-images.githubusercontent.com/104187589/173810494-0f2e7467-802a-4dbd-ae8f-b14342ab748b.png)<br>


**  example 2**<br>
import matplotlib.image as mpimg<br>
import matplotlib.pyplot as plt<br>
img=mpimg.imread('leaf.jpg')<br>
plt.imshow(img)<br>
**  output**<br>
![image](https://user-images.githubusercontent.com/104187589/173809911-dd6192ba-ac88-4f1d-9fa4-05e572091e54.png)<br>


**  example3**<br>
from PIL import Image<br>
img=Image.open("butterfly.jpg")<br>
img=img.rotate(180)<br>
img.show()<br>
cv2.waitKey(0)<br>
cv2.destroyAllwindows()<br>
**  output**<br>
![image](https://user-images.githubusercontent.com/104187589/173813639-77c63461-9337-4783-a3ab-60d7045fb357.png)<br>


** example4**<br>
from PIL import ImageColor<br>
# using getrgb for yellow<br>
img1=ImageColor.getrgb("yellow")<br>
print(img1)<br>
#using getrgb for red<br>
img2=ImageColor.getrgb("red")<br>
print(img2)<br>
**  output**<br>
(255, 255, 0)<br>
(255, 0, 0)<br>

**  example5**<br>
from PIL import Image<br> 
img=Image.new('RGB',(200,400),(255,255,0))<br>
img.show()
**  output**<br>
![image](https://user-images.githubusercontent.com/104187589/173817391-7d4a1af7-ef11-4648-a84a-0c1bf0eb0e40.png)<br>

**  example6**<br>
import cv2<br>
import matplotlib.pyplot as plt<br>
import numpy as np<br>
img=cv2.imread('butterfly.jpg')<br>
plt.imshow(img)<br>
plt.show()<br>
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)<br>
plt.imshow(img)<br>
plt.show()<br>
img=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)<br>
plt.imshow(img)<br>
plt.show()<br>
img=cv2.cvtColor(img,cv2.COLOR_CMYK )<br>
plt.imshow(img)<br>
plt.show()<br>

**  output**<br>
![image](https://user-images.githubusercontent.com/104187589/174037981-b10e816e-a6f0-4bed-bd65-081c3f8a2473.png)<br>
![image](https://user-images.githubusercontent.com/104187589/174038161-41ef87d0-595a-447d-b840-78b98ec7b0d6.png)<br>
![image](https://user-images.githubusercontent.com/104187589/174038393-a6fb31ff-8041-4713-9c2b-abcf07557ecd.png)<br>

**  example 7**<br>
from PIL import Image<br>
image=Image.open('flower.jpg')<br>
print("Filename:",image.filename)<br>
print("Format:",image.format)<br>
print("Mode:",image.mode)<br>
print("Size:",image.size)<br>
print("Width:",image.width)<br>
print("Height:",image.height)<br>
image.close()<br>

**   output **<br>
Filename: flower.jpg<br>
Format: JPEG<br>
Mode: RGB<br>
Size: (474, 474)<br>
Width: 474<br>
Height: 474<br>

** example 8**<br>
import cv2<br>
img=cv2.imread('flower.jpg')<br<>
print('original image length width',img.shape)<br>
cv2.imshow('original image',img)<br>
#to show the resized image<br>
imgresize=cv2.resize(img,(150,160))<br>
cv2.imshow('Resized image',imgresize)<br>
print('Resized image length width',imgresize.shape)<br>
cv2.waitKey(0)<br>

** output**<br>
![image](https://user-images.githubusercontent.com/104187589/174047283-fc01856a-6116-4b59-945a-c62b10eb560f.png)<br>
![image](https://user-images.githubusercontent.com/104187589/174047408-0972ed84-d246-476a-b7c9-b9e9da0a2279.png)<br>
![image](https://user-images.githubusercontent.com/104187589/174047576-971e02bf-902d-4b28-85a1-f7cd23865bfd.png)<br>

**  example 9**<br>
import cv2<br>
# read the image file<br>
img=cv2.imread('flower.jpg')<br>
cv2.imshow("RGB",img)<br>
cv2.waitKey(0)<br>
# Gray scale<br>
img=cv2.imread('flower.jpg',0)<br>
cv2.imshow("Gray",img)<br>
cv2.waitKey(0)<br>
# Binary image<br>
ret,bw_img=cv2.threshold(img,127,255,cv2.THRESH_BINARY)<br>
cv2.imshow("Binary",bw_img)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>

**  output**<br>
![image](https://user-images.githubusercontent.com/104187589/174054528-f0daa172-ec3c-462e-bfe3-730f74bbab44.png)<br>
![image](https://user-images.githubusercontent.com/104187589/174054693-2dc0061f-289a-4c9b-9bda-3273db11b105.png)<br>
![image](https://user-images.githubusercontent.com/104187589/174056336-c10b27d1-bd07-4371-b189-328744b0f156.png)<br>

** 10 Develop a program to readimage using URL**<br>
from skimage import io<br>
import matplotlib.pyplot as plt<br>
url='https://www.thesprucepets.com/thmb/FOLwbR72UrUpF9sZ45RYKzgO8dg=/3072x2034/filters:fill(auto,1)/yellow-tang-fish-508304367-5c3d2790c9e77c000117dcf2.jpg'<br>
image=io.imread(url)<br>
plt.imshow(image)<br>
plt.show()<br>

**  output **<br>
![image](https://user-images.githubusercontent.com/104187589/175007465-0d9b43b2-313e-4440-8b70-22a3e07eea10.png)<br>

** 11 mask and blur the image**<br>
import cv2<br>
import matplotlib.image as mpimg<br>
import matplotlib.pyplot as plt<br>
img=cv2.imread('fish2.jpg')<br>
plt.imshow(img)<br>
plt.show()<br>

**  output**<br>
![image](https://user-images.githubusercontent.com/104187589/175264380-a5c6dff9-d824-4efc-bcd1-9073007e11db.png)<br>

import cv2<br>
hsv_img=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)<br>
light_orange=(1,190,200)<br>
dark_orange=(418,255,255)<br>
mask=cv2.inRange(img,light_orange,dark_orange)<br>
result=cv2.bitwise_and(img,img,mask=mask)<br>
plt.subplot(1,2,1)<br>
plt.imshow(mask,cmap="gray")<br>
plt.subplot(1,2,2)<br>
plt.imshow(result)<br>
plt.show()<br>

**  output**<br>
![image](https://user-images.githubusercontent.com/104187589/175264695-9a14b7e5-7991-4678-92e8-05fecb775404.png)<br>


light_white=(0,0,200)<br>
dark_white=(145,60,255)<br>
mask_white=cv2.inRange(hsv_img,light_white,dark_white)<br>
result_white=cv2.bitwise_and(img,img,mask=mask_white)<br>
plt.subplot(1,2,1)<br>
plt.imshow(mask_white,cmap="gray")<br>
plt.subplot(1,2,2)<br>
plt.imshow(result_white)<br>
plt.show()<br>

** output**<br>
![image](https://user-images.githubusercontent.com/104187589/175264932-5fd19f6a-76ca-4e77-b96f-ac0f0fc02016.png)<br>

final_mask=mask+mask_white<br>
final_result=cv2.bitwise_and(img,img,mask=final_mask)<br>
plt.subplot(1,2,1)<br>
plt.imshow(final_mask,cmap="gray")<br>
plt.subplot(1,2,2)<br>
plt.imshow(final_result)<br>
plt.show()<br>

**  output**<br>
![image](https://user-images.githubusercontent.com/104187589/175265288-7dda1385-4979-426a-a37b-f5d71df2f365.png)<br>

blur=cv2.GaussianBlur(final_result,(7,7),0)<br>
plt.imshow(blur)<br>
plt.show()<br>

**  output**<br>
![image](https://user-images.githubusercontent.com/104187589/175265601-8021868f-10d7-4d3c-995a-44bc183076e3.png)<br>


** 12  perform arithmetic operation on images**<br>
import cv2<br>
import matplotlib.image as mpimg<br>
import matplotlib.pyplot as plt<br>
#Reading image files<br>
img1=cv2.imread('img.jpg')<br>
img2=cv2.imread('imgr.jpg')<br>
# Applying NumPy addition on images<br>
fimg1=img1+img2<br>
plt.imshow(fimg1)<br>
plt.show()<br>
#saving the output image<br>
cv2.imwrite('output.jpg',fimg1)<br>
fimg2=img1-img2<br>
plt.imshow(fimg2)<br>
plt.show()<br>
#saving the output image<br>
cv2.imwrite('output.jpg',fimg2)<br>
fimg3=img1*img2<br>
plt.imshow(fimg3)<br>
plt.show()<br>
#saving the output image<br>
cv2.imwrite('output.jpg',fimg3)<br>
fimg4=img1/img2<br>
plt.imshow(fimg4)<br>
plt.show()<br>
#saving the output image<br>
cv2.imwrite('output.jpg',fimg4)<br>

**  output**<br>
![image](https://user-images.githubusercontent.com/104187589/175271927-7c7cc9bd-34f5-43fe-bc3c-e99a6c4ff7d2.png)<br>
![image](https://user-images.githubusercontent.com/104187589/175272144-3a1e86fd-0a76-42fb-a3fb-5ba83c12bb91.png)<br>
![image](https://user-images.githubusercontent.com/104187589/175272238-dd2f523b-be45-424e-ba9f-7dd77143bc34.png)<br>

** 13 change the images to different color**<br>
import cv2 as c<br>
import numpy as np<br>
from PIL import Image<br>
array = np.zeros( [100,200,3],dtype=np.uint8)<br>
array[:,:100]=[255,130,0]<br>
array[:,100:]=[0,0,255]<br>
img=Image.fromarray(array)<br>
img.save('image1.png')<br>
img.show()<br>
c.waitKey(0)<br>

**  output**<br>
![image](https://user-images.githubusercontent.com/104187589/175282335-b60e4cfe-8e5c-4ae0-ae0b-f5da7b80dc37.png)<br>


** 14 create an image using 2D array**<br>
import cv2<br>
img=cv2.imread('E://butterfly.jpg')<br>
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)<br>
hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)<br>
lab=cv2.cvtColor(img,cv2.COLOR_BGR2LAB)<br>
hls=cv2.cvtColor(img,cv2.COLOR_BGR2HLS)<br>
yuv=cv2.cvtColor(img,cv2.COLOR_BGR2YUV)<br>
cv2.imshow("GRAY image",gray)<br>
cv2.imshow("HSV image",hsv)<br>
cv2.imshow("LAB image",lab)<br>
cv2.imshow("HLS image",hls)<br>
cv2.imshow("YUV image",yuv)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>

**  output**<br>
![image](https://user-images.githubusercontent.com/104187589/175287197-2660046c-c4d3-4dff-96d2-18c724740159.png)<br>
![image](https://user-images.githubusercontent.com/104187589/175287269-23f1ae5b-51e7-486e-82bb-b0113c2cb8bf.png)<br>
![image](https://user-images.githubusercontent.com/104187589/175287371-2a457d95-f19c-47a2-84be-8995c03893fb.png)<br>
![image](https://user-images.githubusercontent.com/104187589/175287457-7f7bcfb0-def7-4504-93a3-66ba43aa0469.png)<br>
![image](https://user-images.githubusercontent.com/104187589/175287557-469a3c5a-a47f-456f-acbf-495b3c4a8adb.png)<br>

** 15 bitwise operation **<br>
import cv2<br>
import matplotlib.pyplot as plt<br>
image1=cv2.imread('plants.jpg',1)<br>
image2=cv2.imread('plants.jpg')<br>
ax=plt.subplots(figsize=(15,10))<br>
bitwiseAnd=cv2.bitwise_and(image1,image2)<br>
bitwiseOr=cv2.bitwise_or(image1,image2)<br>
bitwiseXor=cv2.bitwise_xor(image1,image2)<br>
bitwiseNot_img1=cv2.bitwise_not(image1)<br>
bitwiseNot_img2=cv2.bitwise_not(image2)<br>
plt.subplot(151)<br>
plt.imshow(bitwiseAnd)<br>
plt.subplot(152)<br>
plt.imshow(bitwiseOr)<br>
plt.subplot(153)<br>
plt.imshow(bitwiseXor)<br>
plt.subplot(154)<br>
plt.imshow(bitwiseNot_img1)<br>
plt.subplot(155)<br>
plt.imshow(bitwiseNot_img2)<br>
cv2.waitKey(0)<br>

**  output**<br>
![image](https://user-images.githubusercontent.com/104187589/176410181-a04796fb-f8da-47ca-abe1-84f74c291f91.png)<br>

** 16 blur image**<br>
# importing  libraries<br>
import cv2<br>
import numpy as np<br>
image=cv2.imread('flowers.jpg')<br>
cv2.imshow('original image',image)<br>
cv2.waitKey(0)<br>
# Gaussian Blur<br>
Gaussian=cv2.GaussianBlur(image,(7,7),0)<br>
cv2.imshow('Gaussian Blurring',Gaussian)<br>
cv2.waitKey(0)<br>
# Median Blur<br>
median=cv2.medianBlur(image,5)<br>
cv2.imshow('Median Blurring',median)<br>
cv2.waitKey(0)<br>
#Bilateral Blur<br>
bilateral=cv2.bilateralFilter(image,9,75,75)<br>
cv2.imshow('Bilateral Blurring',bilateral)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>

** output**<br>
![image](https://user-images.githubusercontent.com/104187589/176414183-12d9a17f-c335-483f-88d5-eeba261be867.png)<br>
![image](https://user-images.githubusercontent.com/104187589/176414282-d5b3f541-dd94-4ef5-9bd3-5d809bee64a0.png)<br>
![image](https://user-images.githubusercontent.com/104187589/176414380-e63b2794-ce7b-41e7-bd00-c34323154853.png)<br>
![image](https://user-images.githubusercontent.com/104187589/176414541-5bfe9741-38ae-45d6-81f9-a06437e90895.png)<br>

**  17 Image enhancement**<br>
from PIL import Image<br>
from PIL import ImageEnhance<br>
image=Image.open('butterflys.jpg')<br>
image.show()<br>
enh_bri=ImageEnhance.Brightness(image)<br>
brightness=1.5<br>
image_brightened=enh_bri.enhance(brightness)<br>
image_brightened.show()<br>
enh_col=ImageEnhance.Color(image)<br>
color=1.5<br>
image_colored=enh_col.enhance(color)<br>
image_colored.show()<br>
enh_con=ImageEnhance.Contrast(image)<br>
contrast=1.5<br>
image_contrasted=enh_con.enhance(contrast)<br>
image_contrasted.show()<br>
enh_sha=ImageEnhance.Sharpness(image)<br>
sharpness=3.0<br>
image_sharped=enh_sha.enhance(sharpness)<br>
image_sharped.show()<br>

**  output**<br>
![image](https://user-images.githubusercontent.com/104187589/176418905-239805b2-7d3d-49a0-85b4-f7309db8bbda.png)<br>
![image](https://user-images.githubusercontent.com/104187589/176419017-a78966f9-097a-43ce-b671-573a5d2be9b9.png)<br>
![image](https://user-images.githubusercontent.com/104187589/176419102-be1faad1-c38b-45b0-af69-2b751e1c3d3d.png)<br>
![image](https://user-images.githubusercontent.com/104187589/176419263-9ce7a7da-5475-4f93-8ba2-fdae1abe8fa1.png)<br>
![image](https://user-images.githubusercontent.com/104187589/176419397-b6ea2f75-c18c-4c8d-8352-df76f5f8f907.png)<br>

**  18 Morphological operation**<br>
import cv2<br>
import numpy as np<br>
from matplotlib import pyplot as plt<br>
from PIL import Image,ImageEnhance<br>
img=cv2.imread('butterflys.jpg',0)<br>
ax=plt.subplots(figsize=(20,10))<br>
kernel=np.ones((5,5),np.uint8)<br>
opening=cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)<br>
closing=cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)<br>
erosion=cv2.erode(img,kernel,iterations=1)<br>
dilation=cv2.dilate(img,kernel,iterations=1)<br>
gradient=cv2.morphologyEx(img,cv2.MORPH_GRADIENT,kernel)<br>
plt.subplot(151)<br>
plt.imshow(opening)<br>
plt.subplot(152)<br>
plt.imshow(closing)<br>
plt.subplot(153)<br>
plt.imshow(erosion)<br>
plt.subplot(154)<br>
plt.imshow(dilation)<br>
plt.subplot(155)<br>
plt.imshow(gradient)<br>
cv2.waitKey(0)<br>

**  output**<br>
![image](https://user-images.githubusercontent.com/104187589/176423196-6c867b8e-1ba5-41bc-b750-1280cec5ec0c.png)<br>


19  develop a program to (1) read the image,convert it into grayscale image (2)write(save) the gray scale image and (3)display the original image and gray scale image<br>
import cv2<br>
OriginalImg=cv2.imread('cat.jpg')<br>
GrayImg=cv2.imread('cat.jpg',0)<br>
isSaved=cv2.imwrite('E:/i.jpg',GrayImg)<br>
cv2.imshow('Display Original Image',OriginalImg)<br>
cv2.imshow('Display Grayscale Image',GrayImg)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>
if isSaved:<br>
    print('The Image is successfully saved')<br>
    
**output**<br> 
![image](https://user-images.githubusercontent.com/104187589/179920326-42211e9f-ba76-496a-a890-bdebfbf96101.png)<br>
![image](https://user-images.githubusercontent.com/104187589/179920578-35182aec-46bd-4471-951a-ecd96dd88946.png)<br>
      
20 graylevel slicing with background<br>
import cv2<br>
import numpy as np<br>
from matplotlib import pyplot as plt<br>
image=cv2.imread('flowers.jpg',0)<br>
x,y=image.shape<br>
z=np.zeros((x,y))<br>
for i in range(0,x):<br>
    for j in range(0,y):<br>
        if(image[i][j]>50 and image[i][j]<150):<br>
            z[i][j]=255<br>
        else:<br>
            z[i][j]=image[i][j]<br>
equ=np.hstack((image,z))<br>
plt.title('Graylevel slicing with background')<br>
plt.imshow(equ,'gray')<br>
plt.show()<br>
<br>
 **output**<br>
![image](https://user-images.githubusercontent.com/104187589/179946674-da7d2983-2cb4-4ce8-852d-5847248f404c.png)<br>

21 graylevel slicing without background<br>
import cv2<br>
import numpy as np<br>
from matplotlib import pyplot as plt<br>
image=cv2.imread('flowers.jpg',0)<br>
x,y=image.shape<br>
z=np.zeros((x,y))<br>
for i in range(0,x):<br>
    for j in range(0,y):<br>
        if(image[i][j]>50 and image[i][j]<150):<br>
            z[i][j]=255<br>
        else:<br>
            z[i][j]=0<br>
equ=np.hstack((image,z))<br>
plt.title('Graylevel slicing without background')<br>
plt.imshow(equ,'gray')<br>
plt.show()<br>

**output**<br>
![image](https://user-images.githubusercontent.com/104187589/179947491-c91f9ccb-733c-4c8d-bcce-c732061bbf6a.png)<br>

** 22 intensity transformation (1) image negative (2) log transformation (3)gamma correction**<br>
%matplotlib inline<br>
import imageio<br>
import matplotlib.pyplot as plt<br>
import warnings<br>
import matplotlib.cbook<br>
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)<br>
pic=imageio.imread('cats.jpg')<br>
plt.figure(figsize=(6,6))<br>
plt.imshow(pic);<br>
plt.axis('off');<br>
<br>
**  output**<br>
![image](https://user-images.githubusercontent.com/104187589/179957076-e6c461d7-a37a-48b6-9c36-5283fbd18a7b.png)<br>
<br>
negative=255-pic # neg = (L-1) - img<br>
plt.figure(figsize=(6,6))<br>
plt.imshow(negative);<br>
plt.axis('off');<br>
<br>
![image](https://user-images.githubusercontent.com/104187589/179957289-952c2b50-841d-4646-b303-f17011352fb7.png)<br>
<br>
%matplotlib inline<br>
<br>
import imageio<br>
import numpy as np<br>
import matplotlib.pyplot as plt<br>
<br>
pic=imageio.imread('cats.jpg')<br>
gray=lambda rgb : np.dot(rgb[...,:3],[0.299,0.587,0.114])<br>
gray=gray(pic)<br>
<br>
max_=np.max(gray)<br>

def log_transform():<br>
 return(255/np.log(1+max_))*np.log(1+gray)<br>
plt.figure(figsize=(5,5))<br>
plt.imshow(log_transform(),cmap=plt.get_cmap(name='gray'))<br>
plt.axis('off');<br>
        **  output**<br>
![image](https://user-images.githubusercontent.com/104187589/179957506-564c40ef-4e47-4217-b610-0c2ae73bdeb7.png)<br>
        <br>
        import imageio<br>
import matplotlib.pyplot as plt<br>
<br>
#Gamma encoding<br>
pic=imageio.imread('cats.jpg')<br>
gamma=2.2# Gamma < 1 ~ Dark ; Gamma > 1 ~ Bright<br>
<br>
gamma_correction=((pic/255)**(1/gamma))<br>
plt.figure(figsize=(5,5))<br>
plt.imshow(gamma_correction)<br>
plt.axis('off');<br>
<br>
** output**<br>
![image](https://user-images.githubusercontent.com/104187589/179957693-49fd1560-a7f2-49d3-a1a3-4f26df819aa6.png)<br>
<br>
<br>
**  23 basic image manipulation (1) sharpness (2) flipping (3) cropping**<br>
#Image sharpen<br>
from PIL import Image<br>
from PIL import ImageFilter<br>
import matplotlib.pyplot as plt<br>
# Load the image<br>
my_image=Image.open('cats.jpg')<br>
#Use sharpen function<br>
sharp=my_image.filter(ImageFilter.SHARPEN)<br>
#Save the image<br>
sharp.save('E:/image_sharpen.jpg')<br>
sharp.show()<br>
plt.imshow(sharp)<br>
plt.show()<br>
<br>
**  output**<br>
![image](https://user-images.githubusercontent.com/104187589/179964684-dea3123e-47e1-40de-828a-73c0a87ad41e.png)<br>
<br>
#Image flip<br>
import matplotlib.pyplot as plt<br>
#load the image<br>
img=Image.open('cats.jpg')<br>
plt.imshow(img)<br>
plt.show()<br>
#use the flip function<br>
flip=img.transpose(Image.FLIP_LEFT_RIGHT)<br>
<br>
#save the image<br>
flip.save('E:image_flip.jpg')<br>
plt.imshow(flip)<br>
plt.show()<br>
<br>
**  output**<br>
![image](https://user-images.githubusercontent.com/104187589/179964910-dc8113d3-0217-429c-b9f1-131f449763e1.png)<br>
<br>
# Importing Image class from PIL module<br>
from PIL import Image<br>
import matplotlib.pyplot as plt<br>
# open a image in RGB mode<br>
im=Image.open('cats.jpg')<br>
<br>
# size of the image in pixels(size of original image)<br>
# (This is not mandatory)<br>
width,height=im.size<br>
<br>
#Cropped image of above dimension<br>
# (It will not change original image)<br>
im1=im.crop((25,20,125,120))<br>
<br>
# shows the image in image viewer<br>
im1.show()<br>
plt.imshow(im1)<br>
plt.show()<br>
<br>
**  output**<br>
![image](https://user-images.githubusercontent.com/104187589/179965123-d911361c-6418-4e9d-acde-ee4c13e14b13.png)<br>






















