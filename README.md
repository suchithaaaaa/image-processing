**image-processing**<br>
code for signature verification
from flask import Flask, request, render_template, send_from_directory, jsonify
import sqlite3
from PIL import Image
from Preprocessing import convert_to_image_tensor, invert_image
import torch
from Model import SiameseConvNet, distance_metric
from io import BytesIO
import json
import math


app = Flask(__name__, static_folder='./frontend/build/static', template_folder='./frontend/build')

def load_model():
    device = torch.device('cpu')
    model = SiameseConvNet().eval()
    model.load_state_dict(torch.load('Models/model_large_epoch_20', map_location=device))
    return model


def connect_to_db():
    conn = sqlite3.connect('user_signatures.db')
    return conn


def get_file_from_db(customer_id):
    cursor = connect_to_db().cursor()
    select_fname = """SELECT sign1,sign2,sign3 from signatures where customer_id = ?"""
    cursor.execute(select_fname, (customer_id,))
    item = cursor.fetchone()
    cursor.connection.commit()
    return item


def main():
    CREATE_TABLE = """CREATE TABLE IF NOT EXISTS signatures (customer_id TEXT PRIMARY KEY,sign1 BLOB, sign2 BLOB, sign3 BLOB)"""
    cursor = connect_to_db().cursor()
    cursor.execute(CREATE_TABLE)
    cursor.connection.commit()
    # For heroku, remove this line. We'll use gunicorn to run the app
    app.run() # app.run(debug=True) 

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file1 = request.files['uploadedImage1']
    file2 = request.files['uploadedImage2']
    file3 = request.files['uploadedImage3']
    customer_id = request.form['customerID']
    print(customer_id)
    try:
        conn = connect_to_db()
        cursor = conn.cursor()
        query = """DELETE FROM signatures where customer_id=?"""
        cursor.execute(query, (customer_id,))
        cursor = conn.cursor()
        query = """INSERT INTO signatures VALUES(?,?,?,?)"""
        cursor.execute(query, (customer_id, file1.read(), file2.read(), file3.read()))
        conn.commit()
        return jsonify({"error": False})
    except Exception as e:
        print(e)
        return jsonify({"error": True})

@app.route('/verify', methods=['POST'])
def verify():
    try:
        customer_id = request.form['customerID']
        input_image = Image.open(request.files['newSignature'])
        input_image_tensor = convert_to_image_tensor(invert_image(input_image)).view(1,1,220,155)
        customer_sample_images = get_file_from_db(customer_id)
        if not customer_sample_images:
            return jsonify({'error':True})
        anchor_images = [Image.open(BytesIO(x)) for x in customer_sample_images]
        anchor_image_tensors = [convert_to_image_tensor(invert_image(x)).view(-1, 1, 220, 155) 
                        for x in anchor_images]
        model = load_model()
        mindist = math.inf
        for anci in anchor_image_tensors:
            f_A, f_X = model.forward(anci, input_image_tensor)
            dist = float(distance_metric(f_A, f_X).detach().numpy())
            mindist = min(mindist, dist)

            if dist <= 0.145139:  # Threshold obtained using Test.py
                return jsonify({"match": True, "error": False, "threshold":"%.6f" % (0.145139), "distance":"%.6f"%(mindist)})
        return jsonify({"match": False, "error": False, "threshold":0.145139, "distance":round(mindist, 6)})
    except Exception as e:
        print(e)
        return jsonify({"error":True})

@app.route("/manifest.json")
def manifest():
    return send_from_directory('./frontend/build', 'manifest.json')

@app.route("/favicon.ico")
def favicon():
    return send_from_directory('./frontend/build', 'favicon.ico')

if __name__=='__main__':
    
    main()





<br>
1 **Develop a program to display grayscale image using read and write operation**<br>
import cv2<br>
img=cv2.imread('plants.jpg',0)<br>
cv2.imshow('image',img)<br>
cv2.waitKey(0)<br>
cv2.destroyAllWindows()<br>
<br>
**output**<br>
![image](https://user-images.githubusercontent.com/104187589/173810494-0f2e7467-802a-4dbd-ae8f-b14342ab748b.png)<br>
<br>
2 **Develpo a program to display the image using matplotlib**<br>
import matplotlib.image as mpimg<br>
import matplotlib.pyplot as plt<br>
img=mpimg.imread('leaf.jpg')<br>
plt.imshow(img)<br>
**output**<br>
![image](https://user-images.githubusercontent.com/104187589/173809911-dd6192ba-ac88-4f1d-9fa4-05e572091e54.png)<br>
<br>
<br>
3 **Develop a program  to perform linear transformation**<br>
from PIL import Image<br>
img=Image.open("butterfly.jpg")<br>
img=img.rotate(180)<br>
img.show()<br>
cv2.waitKey(0)<br>
cv2.destroyAllwindows()<br>
**output**<br>
![image](https://user-images.githubusercontent.com/104187589/173813639-77c63461-9337-4783-a3ab-60d7045fb357.png)<br>
<br>
<br>
4 **develop a program to convert color string 10 RGB color value**
from PIL import ImageColor<br>
# using getrgb for yellow<br>
img1=ImageColor.getrgb("yellow")<br>
print(img1)<br>
#using getrgb for red<br>
img2=ImageColor.getrgb("red")<br>
print(img2)<br>
<br>
**output**<br>
(255, 255, 0)<br>
(255, 0, 0)<br>
<br>
5 **write a program to create Image using color**<br>
from PIL import Image<br> 
img=Image.new('RGB',(200,400),(255,255,0))<br>
img.show()
**output**<br>
![image](https://user-images.githubusercontent.com/104187589/173817391-7d4a1af7-ef11-4648-a84a-0c1bf0eb0e40.png)<br>
<br>
6**develop a program to visualize the image using various color spaces**<br>
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
<br>
**output**<br>
![image](https://user-images.githubusercontent.com/104187589/174037981-b10e816e-a6f0-4bed-bd65-081c3f8a2473.png)<br>
![image](https://user-images.githubusercontent.com/104187589/174038161-41ef87d0-595a-447d-b840-78b98ec7b0d6.png)<br>
![image](https://user-images.githubusercontent.com/104187589/174038393-a6fb31ff-8041-4713-9c2b-abcf07557ecd.png)<br>
<br>
7 **display the image attributes**<br>
from PIL import Image<br>
image=Image.open('flower.jpg')<br>
print("Filename:",image.filename)<br>
print("Format:",image.format)<br>
print("Mode:",image.mode)<br>
print("Size:",image.size)<br>
print("Width:",image.width)<br>
print("Height:",image.height)<br>
image.close()<br>
<br>
**output **<br>
Filename: flower.jpg<br>
Format: JPEG<br>
Mode: RGB<br>
Size: (474, 474)<br>
Width: 474<br>
Height: 474<br>
<br>
8 **resize the image**<br>
import cv2<br>
img=cv2.imread('flower.jpg')<br<>
print('original image length width',img.shape)<br>
cv2.imshow('original image',img)<br>
#to show the resized image<br>
imgresize=cv2.resize(img,(150,160))<br>
cv2.imshow('Resized image',imgresize)<br>
print('Resized image length width',imgresize.shape)<br>
cv2.waitKey(0)<br>
<br>
**output**<br>
![image](https://user-images.githubusercontent.com/104187589/174047283-fc01856a-6116-4b59-945a-c62b10eb560f.png)<br>
![image](https://user-images.githubusercontent.com/104187589/174047408-0972ed84-d246-476a-b7c9-b9e9da0a2279.png)<br>
![image](https://user-images.githubusercontent.com/104187589/174047576-971e02bf-902d-4b28-85a1-f7cd23865bfd.png)<br>

9 **convert the image gray to binary**<br>
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

**output**<br>
![image](https://user-images.githubusercontent.com/104187589/174054528-f0daa172-ec3c-462e-bfe3-730f74bbab44.png)<br>
![image](https://user-images.githubusercontent.com/104187589/174054693-2dc0061f-289a-4c9b-9bda-3273db11b105.png)<br>
![image](https://user-images.githubusercontent.com/104187589/174056336-c10b27d1-bd07-4371-b189-328744b0f156.png)<br>
<br>
10 **Develop a program to readimage using URL**<br>
from skimage import io<br>
import matplotlib.pyplot as plt<br>
url='https://www.thesprucepets.com/thmb/FOLwbR72UrUpF9sZ45RYKzgO8dg=/3072x2034/filters:fill(auto,1)/yellow-tang-fish-508304367-5c3d2790c9e77c000117dcf2.jpg'<br>
image=io.imread(url)<br>
plt.imshow(image)<br>
plt.show()<br>
<br>
**output **<br>
![image](https://user-images.githubusercontent.com/104187589/175007465-0d9b43b2-313e-4440-8b70-22a3e07eea10.png)<br>
<br>
11 **mask and blur the image**<br>
import cv2<br>
import matplotlib.image as mpimg<br>
import matplotlib.pyplot as plt<br>
img=cv2.imread('fish2.jpg')<br>
plt.imshow(img)<br>
plt.show()<br>
<br>
**output**<br>
![image](https://user-images.githubusercontent.com/104187589/175264380-a5c6dff9-d824-4efc-bcd1-9073007e11db.png)<br>
<br>
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
<br>
**output**<br>
![image](https://user-images.githubusercontent.com/104187589/175264695-9a14b7e5-7991-4678-92e8-05fecb775404.png)<br>
<br>
light_white=(0,0,200)<br>
dark_white=(145,60,255)<br>
mask_white=cv2.inRange(hsv_img,light_white,dark_white)<br>
result_white=cv2.bitwise_and(img,img,mask=mask_white)<br>
plt.subplot(1,2,1)<br>
plt.imshow(mask_white,cmap="gray")<br>
plt.subplot(1,2,2)<br>
plt.imshow(result_white)<br>
plt.show()<br>
<br>
**output**<br>
![image](https://user-images.githubusercontent.com/104187589/175264932-5fd19f6a-76ca-4e77-b96f-ac0f0fc02016.png)<br>
<br>
final_mask=mask+mask_white<br>
final_result=cv2.bitwise_and(img,img,mask=final_mask)<br>
plt.subplot(1,2,1)<br>
plt.imshow(final_mask,cmap="gray")<br>
plt.subplot(1,2,2)<br>
plt.imshow(final_result)<br>
plt.show()<br>
<br>
**output**<br>
![image](https://user-images.githubusercontent.com/104187589/175265288-7dda1385-4979-426a-a37b-f5d71df2f365.png)<br>
blur=cv2.GaussianBlur(final_result,(7,7),0)<br>
plt.imshow(blur)<br>
plt.show()<br>

**output**<br>
![image](https://user-images.githubusercontent.com/104187589/175265601-8021868f-10d7-4d3c-995a-44bc183076e3.png)<br>
<br>
12 **perform arithmetic operation on images**<br>
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
<br>
**output**<br>
![image](https://user-images.githubusercontent.com/104187589/175271927-7c7cc9bd-34f5-43fe-bc3c-e99a6c4ff7d2.png)<br>
![image](https://user-images.githubusercontent.com/104187589/175272144-3a1e86fd-0a76-42fb-a3fb-5ba83c12bb91.png)<br>
![image](https://user-images.githubusercontent.com/104187589/175272238-dd2f523b-be45-424e-ba9f-7dd77143bc34.png)<br>
<br>
13 **change the images to different color**<br>
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
<br>
**output**<br>
![image](https://user-images.githubusercontent.com/104187589/175282335-b60e4cfe-8e5c-4ae0-ae0b-f5da7b80dc37.png)<br>
<br>
14 **create an image using 2D array**<br>
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
<br>
**output**<br>
![image](https://user-images.githubusercontent.com/104187589/175287197-2660046c-c4d3-4dff-96d2-18c724740159.png)<br>
![image](https://user-images.githubusercontent.com/104187589/175287269-23f1ae5b-51e7-486e-82bb-b0113c2cb8bf.png)<br>
![image](https://user-images.githubusercontent.com/104187589/175287371-2a457d95-f19c-47a2-84be-8995c03893fb.png)<br>
![image](https://user-images.githubusercontent.com/104187589/175287457-7f7bcfb0-def7-4504-93a3-66ba43aa0469.png)<br>
![image](https://user-images.githubusercontent.com/104187589/175287557-469a3c5a-a47f-456f-acbf-495b3c4a8adb.png)<br>
<br>
15 **bitwise operation **<br>
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
<br>
**output**<br>
![image](https://user-images.githubusercontent.com/104187589/176410181-a04796fb-f8da-47ca-abe1-84f74c291f91.png)<br>
<br>
16 **blur image**<br>
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
<br>
**output**<br>
![image](https://user-images.githubusercontent.com/104187589/176414183-12d9a17f-c335-483f-88d5-eeba261be867.png)<br>
![image](https://user-images.githubusercontent.com/104187589/176414282-d5b3f541-dd94-4ef5-9bd3-5d809bee64a0.png)<br>
![image](https://user-images.githubusercontent.com/104187589/176414380-e63b2794-ce7b-41e7-bd00-c34323154853.png)<br>
![image](https://user-images.githubusercontent.com/104187589/176414541-5bfe9741-38ae-45d6-81f9-a06437e90895.png)<br>
<br>
17 **Image enhancement**<br>
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

**output**<br>
![image](https://user-images.githubusercontent.com/104187589/176418905-239805b2-7d3d-49a0-85b4-f7309db8bbda.png)<br>
![image](https://user-images.githubusercontent.com/104187589/176419017-a78966f9-097a-43ce-b671-573a5d2be9b9.png)<br>
![image](https://user-images.githubusercontent.com/104187589/176419102-be1faad1-c38b-45b0-af69-2b751e1c3d3d.png)<br>
![image](https://user-images.githubusercontent.com/104187589/176419263-9ce7a7da-5475-4f93-8ba2-fdae1abe8fa1.png)<br>
![image](https://user-images.githubusercontent.com/104187589/176419397-b6ea2f75-c18c-4c8d-8352-df76f5f8f907.png)<br>
<br>
18 **Morphological operation**<br>
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
<br>
**output**<br>
![image](https://user-images.githubusercontent.com/104187589/176423196-6c867b8e-1ba5-41bc-b750-1280cec5ec0c.png)<br>
<br>
19  **develop a program to (1) read the image,convert it into grayscale image (2)write(save) the gray scale image and (3)display the original image and gray scale image**<br>
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
 <br>   
**output**<br> 
![image](https://user-images.githubusercontent.com/104187589/179920326-42211e9f-ba76-496a-a890-bdebfbf96101.png)<br>
![image](https://user-images.githubusercontent.com/104187589/179920578-35182aec-46bd-4471-951a-ecd96dd88946.png)<br>
     <br> 
20 **graylevel slicing with background**<br>
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

21 **graylevel slicing without background**<br>
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

22 **intensity transformation (1) image negative (2) log transformation (3)gamma correction**<br>
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
**output**<br>
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
**output**<br>
![image](https://user-images.githubusercontent.com/104187589/179957693-49fd1560-a7f2-49d3-a1a3-4f26df819aa6.png)<br>
<br>
<br>
23 **basic image manipulation (1) sharpness (2) flipping (3) cropping**<br>
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
**output**<br>
![image](https://user-images.githubusercontent.com/104187589/179965123-d911361c-6418-4e9d-acde-ee4c13e14b13.png)<br>

**1  standard deviation**  <br>
from PIL import Image, ImageStat<br>

im = Image.open('butterflys.jpg')<br>
stat = ImageStat.Stat(im)<br>
print(stat.stddev)<br>
<br>
**output**<br>
[76.55933208003023, 57.05839133604739, 55.21102658091532]<br>
<br>
**maximum**<br>
import cv2<br>
import numpy as np<br>
img=cv2.imread('butterflys.jpg')<br>
cv2.imshow('butterflys.jpg',img)<br>
cv2.waitKey(0)<br>
#max_channels=np.amax([np.amax(img[:,:,0]),np.amax(img[:,:,1]),np.amax(img[:,:,2])])<br>
#print(max_channels)<br>
np.max(img)<br>
**output**<br>
![image](https://user-images.githubusercontent.com/104187589/181221228-835e8fb1-a3d7-43fe-96f5-e1ff38768d53.png)<br>
<br>
**minimum**<br>
import cv2<br>
import numpy as np<br>
img=cv2.imread('butterflys.jpg')<br>
cv2.imshow('butterfys.jpg',img)<br>
cv2.waitKey(0)<br>
#min_channels=np.amin([np.amin(img[:,:,0]),np.amin(img[:,:,1]),np.amin(img[:,:,2])])<br>
#print(min_channels)<br>
np.min(img)<br>
<br>
**output**<br>
![image](https://user-images.githubusercontent.com/104187589/181225007-28c73f50-1715-422c-b57f-d754da2c27de.png)<br>
<br>
**average**<br>
import cv2<br>
import numpy as np<br>
img=cv2.imread('butterflys.jpg')<br>
cv2.imshow('butterflys.jpg',img)<br>
cv2.waitKey(0)<br>
np.average(img)<br>
<br>
**output**<br>
89.87986752773638<br>
<br>
** standard deviation**<br>
import cv2<br>
import numpy as np<br>
img=cv2.imread('butterflys.jpg')<br>
cv2.imshow('butterflys.jpg',img)<br>
cv2.waitKey(0)<br>
np.std(img)<br>
<br>
**output**<br>
67.00845599929046<br>
<br>
**matrix**<br>
from PIL import Image<br>
from numpy import asarray<br>
img = Image.open('flowers.jpg')<br>
numpydata = asarray(img)<br>
print(numpydata)<br>
**output**<br>
[[ 62 111  45]<br>
  [ 63 112  46]<br>
  [ 61 112  46]<br>
  ...<br>
  [ 72  98  35]<br>
  [ 75 101  38]<br>
  [ 77 103  40]]<br>
<br>
 [[ 65 114  48]<br>
  [ 68 117  51]<br>
  [ 63 112  47]<br>
  ...<br>
  [ 71  99  38]<br>
  [ 74 103  39]<br>
  [ 78 107  43]]<br>
<br>
 [[ 62 111  45]<br>
  [ 71 120  54]<br>
  [ 69 118  52]<br>
  ...<br>
  [ 73 105  42]<br>
  [ 78 108  46]<br>
  [ 85 115  53]]<br>
<br>
 ...<br>
<br>
 [[ 72  99  32]<br>
  [ 71  98  31]<br>
  [ 67  94  27]<br>
  ...<br>
  [ 63 101  50]<br>
  [ 67 105  54]<br>
  [ 48  86  35]]<br>
<br>
 [[ 66  92  21]<br>
  [ 69  95  24]<br>
  [ 69  96  25]<br>
  ...<br>
  [ 53  91  40]<br>
  [ 65 103  52]<br>
  [ 77 115  64]]<br>
<br>
 [[ 68  94  21]<br>
  [ 68  94  21]<br>
  [ 65  92  21]<br>
  ...<br>
  [ 54  92  41]<br>
  [ 53  90  39]<br>
  [ 53  90  39]]]<br>
  <br>
  <br>
  from PIL import Image<br>
import matplotlib.pyplot as plt<br>
input_image = Image.new(mode="RGB", size=(1000, 1000),color="pink")<br>
pixel_map = input_image.load()<br>
width, height = input_image.size<br>
z = 100<br>
for i in range(width):<br>
    for j in range(height):<br>
          if((i >= z and i <= width-z) and (j >= z and j <= height-z)):<br>
            pixel_map[i, j] = (230,230,250)<br>
else:<br>
     pixel_map[i, j] = (216,191,216)<br>
for i in range(width):<br>
    pixel_map[i, i] = (0, 0, 255)<br>
    pixel_map[i, width-i-1] = (0, 0, 255)<br>
plt.imshow(input_image)<br>
plt.show()<br>
**output**<br>
![image](https://user-images.githubusercontent.com/104187589/181229615-20853161-770f-4593-b01d-bf0e47b9501f.png)<br>
<br>
<br>
import numpy as np<br>
import matplotlib.pyplot as plt<br>
arr = np.zeros((256,256,3), dtype=np.uint8)<br>
imgsize = arr.shape[:2]<br>
innerColor = (255, 255, 255)<br>
outerColor = (0, 0, 0)<br>
for y in range(imgsize[1]):<br>
    for x in range(imgsize[0]):<br>
        distanceToCenter = np.sqrt((x - imgsize[0]//2) ** 2 + (y - imgsize[1]//2) ** 2)<br>
        distanceToCenter = distanceToCenter / (np.sqrt(2) * imgsize[0]/2)<br>
        r = outerColor[0] * distanceToCenter + innerColor[0] * (1 - distanceToCenter)<br>
        g = outerColor[1] * distanceToCenter + innerColor[1] * (1 - distanceToCenter)<br>
        b = outerColor[2] * distanceToCenter + innerColor[2] * (1 - distanceToCenter)<br>
        arr[y, x] = (int(r), int(g), int(b))<br>
plt.imshow(arr, cmap='gray')<br>
plt.show()<br>
**output**<br>
![image](https://user-images.githubusercontent.com/104187589/181229854-f25046e6-923b-44c6-8eb8-61461f5e36be.png)<br>
<br>
<br>
import numpy as np<br>
import matplotlib.pyplot as plt<br>
<br>
imgsize=(650,650)<br>
image = Image.new('RGB', imgsize)<br>
innerColor = [153,0,0]<br>
for y in range(imgsize[1]):<br>
    for x in range(imgsize[0]):<br>
        distanceToCenter =np.sqrt((x - imgsize[0]/2) ** 2 + (y - imgsize[1]/2) ** 2)<br>
        distanceToCenter = (distanceToCenter) / (np.sqrt(2) * imgsize[0]/2)<br>
        r = distanceToCenter + innerColor[0] * (1 - distanceToCenter)<br>
        g = distanceToCenter + innerColor[1] * (1 - distanceToCenter)<br>
        image.putpixel((x, y), (int(r), int(g), int(b)))<br>
plt.imshow(image)<br>
plt.show()<br>
**output**<br>
![image](https://user-images.githubusercontent.com/104187589/181230362-71e74f24-5365-458c-978f-d7903496c18b.png)<br>
<br>
<br>
from PIL import Image<br>
import numpy as np<br>
import matplotlib.pyplot as plt<br>
w, h = 512, 512<br>
data = np.zeros((h, w, 3), dtype=np.uint8)<br>
data[0:100, 0:100] = [255, 0, 0]<br>
data[100:200, 100:200] = [255, 0, 255]<br>
data[200:300, 200:300] = [0, 255, 0]<br>
data[300:400, 300:400] = [255, 255, 0]<br>
data[400:500, 400:500] = [0, 255, 255]<br>
img = Image.fromarray(data, 'RGB')<br>
img.save('flowers.jpg')<br>
plt.imshow(img)<br>
plt.show()<br>
<br>
**output**<br>
![image](https://user-images.githubusercontent.com/104187589/181230617-817756d1-9992-40d8-8980-0e10c41e8116.png)<br>
<br>
**matrix form**<br>
#Python3 program for printing<br>
#the rectangular pattern<br>
 <br>
**Function to print the pattern**<br>
def printPattern(n):<br>
 <br>
    arraySize = n * 2 - 1;<br>
    result = [[0 for x in range(arraySize)]<br>
                 for y in range(arraySize)];<br>
         <br>
    #Fill the values<br>
    for i in range(arraySize):<br>
        for j in range(arraySize):<br>
            if(abs(i - (arraySize // 2)) ><br>
               abs(j - (arraySize // 2))):<br>
                result[i][j] = abs(i - (arraySize // 2));<br>
            else:<br>
                result[i][j] = abs(j - (arraySize // 2));<br>
             <br>
    # Print the array<br>
    for i in range(arraySize):<br>
        for j in range(arraySize):<br>
            print(result[i][j], end = " ");<br>
        print("");<br>
 <br>
#Driver Code<br>
n = 4;<br>
 <br>
printPattern(n);<br>
<br>
**output**<br>
3 3 3 3 3 3 3 <br>
3 2 2 2 2 2 3 <br>
3 2 1 1 1 2 3 <br>
3 2 1 0 1 2 3 <br>
3 2 1 1 1 2 3 <br>
3 2 2 2 2 2 3 <br>
3 3 3 3 3 3 3<br>
<br>
**image displayed in matrix representation**<br>
**First import the required Python Libraries**<br>
import numpy as np<br>
import matplotlib.pyplot as plt<br>
from skimage import img_as_uint<br>
from skimage.io import imshow, imread<br>
from skimage.color import rgb2hsv<br>
from skimage.color import rgb2gray<br>
array_1 = np.array([[255, 0,17], <br>
                    [100,0, 255],<br>
                    [255,0,35]])<br>
imshow(array_1,cmap='gray');<br>
<br>
**output**<br>
![image](https://user-images.githubusercontent.com/104187589/181437231-a369c244-9ca9-4ac0-82f9-513b268154f7.png)<br>
<br><br>
**First import the required Python Libraries**<br>
import numpy as np<br>
import matplotlib.pyplot as plt<br>
from skimage import img_as_uint<br>
from skimage.io import imshow, imread<br>
from skimage.color import rgb2hsv<br>
from skimage.color import rgb2gray<br>
array_1 = np.array([[255, 0,17], <br>
                    [100,0, 255],<br>
                    [255,0,35]])<br>
imshow(array_1,cmap='gray');<br>
![image](https://user-images.githubusercontent.com/104187589/181437602-6b5c4781-eb26-47de-bbcd-295d1034bba9.png)<br>
<br>
**matrix to display image with max,min,average and standard deviation**<br>
import numpy as np<br>
import matplotlib.pyplot as plt<br>
array_colors = np.array([[[245, 20, 36],<br> 
                         [10, 215, 30],<br>
                         [40, 50, 205]],<br>
                         [[70, 50, 10], <br>
                    [25, 230, 85],<br>
                    [12, 128, 128]],<br>
                    [[25, 212, 3], <br>
                    [55, 5, 250],<br>
                    [240, 152, 25]],<br>
                    ])<br>
plt.imshow(array_colors)<br>
np.max(array_colors)<br>
**output**<br>
![image](https://user-images.githubusercontent.com/104187589/181444083-8150b23d-112d-44f6-8621-4709f4b367d3.png)<br>
<br>
import numpy as np<br>
import matplotlib.pyplot as plt<br>
<br>
array_colors = np.array([[[245, 20, 36], <br>
                         [10, 215, 30],<br>
                         [40, 50, 205]],<br>
                         [[70, 50, 10], <br>
                    [25, 230, 85],<br>
                    [12, 128, 128]],<br>
                    [[25, 212, 3], <br>
                    [55, 5, 250],<br>
                    [240, 152, 25]],<br>
                    ])<br>
plt.imshow(array_colors)<br>
np.min(array_colors)<br>
<br>
**output**<br>
![image](https://user-images.githubusercontent.com/104187589/181444377-36f90f16-ef44-495f-bdf9-a6e9c8578fc1.png)<br>
import numpy as np<br>
import matplotlib.pyplot as plt<br>
array_colors = np.array([[[245, 20, 36], <br>
                         [10, 215, 30],<br>
                         [40, 50, 205]],<br>
                         [[70, 50, 10], <br>
                    [25, 230, 85],<br>
                    [12, 128, 128]],<br>
                    [[25, 212, 3], <br>
                    [55, 5, 250],<br>
                    [240, 152, 25]],<br>
                    ])<br>
plt.imshow(array_colors)<br>
np.average(array_colors)<br>
<br>
![image](https://user-images.githubusercontent.com/104187589/181444626-c347b067-a178-480d-9aa5-a29fc6bcba0d.png)<br>
<br>
import numpy as np<br>
import matplotlib.pyplot as plt<br>
array_colors = np.array([[[245, 20, 36],<br> 
                         [10, 215, 30],<br>
                         [40, 50, 205]],<br>
                         [[70, 50, 10], <br>
                    [25, 230, 85],<br>
                    [12, 128, 128]],<br>
                    [[25, 212, 3], <br>
                    [55, 5, 250],<br>
                    [240, 152, 25]],<br>
                    ])<br>
plt.imshow(array_colors)<br>
np.std(array_colors)<br>
<br>
![image](https://user-images.githubusercontent.com/104187589/181444827-ac207970-89f9-40e0-8861-885bf002e344.png)<br>
<br>





























