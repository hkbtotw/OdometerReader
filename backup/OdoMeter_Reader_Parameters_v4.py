import os
import sys
import requests
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import Polygon
from PIL import Image,  ImageEnhance
from io import BytesIO
import cv2
import glob
import pandas as pd
import numpy as np
import re
from PIL import Image, ExifTags, ImageDraw
from scipy.ndimage.filters import median_filter

# Mac Endpoint (Use if TT enpoint is out of limit at certain period of time due to free plan selection)
mac_ep2="https://southeastasia.api.cognitive.microsoft.com/"
mac_sub2='f34ac0f710884d41a55d6385545c2962'

# Meter Detection
dm_endpoint="https://eastus.api.cognitive.microsoft.com/"
subscription='4dcf0fb76847411382a3983c87a82e7c'

#dm_ocr_url = dm_endpoint + "customvision/v3.0/Prediction/37ef4479-4819-4b1e-b16f-7a2f24e9bfb2/detect/iterations/Iteration1/image"
dm_ocr_url = dm_endpoint + "customvision/v3.0/Prediction/37ef4479-4819-4b1e-b16f-7a2f24e9bfb2/detect/iterations/Iteration10/image"

# Digit recognition
dc_ocr_url = dm_endpoint + "customvision/v3.0/Prediction/eade7071-a13c-4eab-9f81-0b4afeaa8be9/detect/iterations/Iteration2/image"


# Digit location
mac_endpoint="https://eastus.api.cognitive.microsoft.com/"
mac_ocr_url = mac_endpoint + "customvision/v3.0/Prediction/b62e724e-6590-4317-8424-32f96223f8dd/detect/iterations/Iteration35/image"

#Semi-Position recognition by Classfication
mac_Semi_cl_ocr_url = mac_endpoint + "customvision/v3.0/Prediction/7b99cf70-06ec-45e9-86f1-f2e2d320350b/classify/iterations/Iteration2/image"

#Full+Semi recognition by Classfication
mac_FSemi_cl_ocr_url = mac_endpoint + "customvision/v3.0/Prediction/433bd650-ef35-48f6-aeb0-e9b7f4071f2b/classify/iterations/Iteration2/image"

#Gray_Full+Semi recognition by Classfication
mac_GFS_cl_ocr_url = mac_endpoint + "customvision/v3.0/Prediction/9f605dd7-932e-41a5-8ed3-461f7c55d50c/classify/iterations/Iteration2/image"

# Digit recognition by Classification

mac_cl_ocr_url = mac_endpoint + "customvision/v3.0/Prediction/f48ba09e-7382-483c-81cf-cf04415949a8/classify/iterations/Iteration11/image"

mac_subscription='4dcf0fb76847411382a3983c87a82e7c'
mac_headers = {'Prediction-Key': mac_subscription, 'Content-Type': 'application/octet-stream'}

#Read API
read_ocr_url = dm_endpoint +"vision/v2.0/read/core/asyncBatchAnalyze"

#Header
headers = {'Prediction-Key': subscription, 'Content-Type': 'application/octet-stream'}

# Compute Average color of the image
def averagecolor(myimg):
    avg_color_per_row = np.average(myimg, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)
    #print('===>',avg_color)
    return avg_color

# Check if the cropped image has mostly white area (License plat tag)
def IfLicensePlatTag(img1):
    avgval=averagecolor(np.array(img1))

    dummy=0
    dummy_avg=0
    color_threshold=150
    for cor in range(3):
        #print(cor,' : ',avgval[cor])
        dummy_avg=dummy_avg+avgval[cor]
        #if(avgval[cor]>color_threshold):
        #    dummy=dummy+3

    if(dummy_avg/3.0>color_threshold):
        dummy=1
    #print(' ==> ',dummy, ' :: ',dummy_avg/3.0)
    return dummy

# Mask the cropped image to Gray (Better than Black)
def MaskArea(area, filepath):    
    #print(' Mask Gray ')
    #print(' ==> ',area)
    start_point = (area[2], area[0]) 
    end_point = (area[3],area[1]) 
   
    # Gray color in BGR 
    color = (128, 128, 128) 
   
    # Line thickness of -1 px 
    # Thickness of -1 will fill the entire shape 
    thickness = -1
   
    # Using cv2.rectangle() method 
    # Draw a rectangle of black color of thickness -1 px 
    opencv_image=cv2.imread(filepath) # open image using openCV2
    image = cv2.rectangle(opencv_image, start_point, end_point, color, thickness)
    #cv2.imwrite(path_out,image)
    img1, img_m, img_dd, area, prob=Detect_Meter_2(image,dm_ocr_url,headers)
    return img1, img_m, img_dd, area, image

def MaskArea_2(area, img_in):    
    #print(' Mask Gray 2 ')
    #print(' ==> ',area)
    start_point = (area[2], area[0]) 
    end_point = (area[3],area[1]) 
   
    # Gray color in BGR 
    color = (128, 128, 128) 
   
    # Line thickness of -1 px 
    # Thickness of -1 will fill the entire shape 
    thickness = -1
   
    image = cv2.rectangle(np.array(img_in), start_point, end_point, color, thickness)
    img1, img_m, img_dd, area, prob=Detect_Meter_2(image,dm_ocr_url,headers)
    return img1, img_m, img_dd, area

def rescale_image(img):
    #Getting the bigger side of the image
    s = max(img.shape[0:2])
    #Creating a dark square with NUMPY  
    f = np.zeros((2*s,2*s),np.uint8)

    #Getting the centering position
    ax,ay = (s - img.shape[1])//2,(s - img.shape[0])//2

    #Pasting the 'image' in a centering position
    f[ay:img.shape[0]+ay,ax:ax+img.shape[1]] = img
    return f

def Number_Reader_ReadAPI_3(image, ocr_url, subscription):

    nx, ny = image.size
    im2 = image.resize((int(nx*0.7), int(ny*0.7)), Image.BICUBIC)
    #im2 = image.resize((int(nx*1.0), int(ny*1.0)), Image.BICUBIC)

    nx2, ny2 = im2.size
    
    im_cv = np.array(im2)
    #img1 = cv2.cvtColor(im_cv, cv2.COLOR_BGR2RGB)
    img1 = cv2.cvtColor(np.array(im2), cv2.COLOR_BGR2GRAY)
    #dst2 = cv2.cvtColor(np.array(im2), cv2.COLOR_BGR2GRAY)
    img = cv2.fastNlMeansDenoising(img1,None,6,9,21)
    #dst = cv2.fastNlMeansDenoising(np.array(im2),None,3,7,21)
    #dst2 = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    #(thresh, dst2) = cv2.threshold(dst, 105, 255, cv2.THRESH_BINARY)

    #dst2 = cv2.fastNlMeansDenoising(dst,None,6,7,21)
    #medfilter=median_filter(img2,1)
    #lap=cv2.Laplacian(medfilter,cv2.CV_64F)
    #img=medfilter-0.01*lap

    #img3 = cv2.medianBlur(img2,5)
    #img = cv2.adaptiveThreshold(img3,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    #print(' img type: ',type(img), ' ==> ',img.shape)
    imgin=rescale_image(img)

    im_resize=imgin
    is_success, im_buf_arr = cv2.imencode(".jpg", im_resize)
    byte_im = im_buf_arr.tobytes()
   
    image_data=byte_im

    headers = {'Ocp-Apim-Subscription-Key': subscription, 'Content-Type': 'application/octet-stream'}
    params = {'language': 'en', 'detectOrientation': 'true'}
    #params = {'language': 'unk'}

    response = requests.post(ocr_url, headers=headers, params=params, data = image_data)

    response.raise_for_status()

    #==============================

    # Holds the URI used to retrieve the recognized text.
    operation_url = response.headers["Operation-Location"]

    # The recognized text isn't immediately available, so poll to wait for completion.
    analysis = {}
    poll = True
    while (poll):
        response_final = requests.get(
            response.headers["Operation-Location"], headers=headers)
        analysis = response_final.json()
        #print(analysis)
        time.sleep(1)
        if ("recognitionResults" in analysis):
            poll = False
        if ("status" in analysis and analysis['status'] == 'Failed'):
            poll = False

    # Extract the word bounding boxes and text.
    line_infos = [line["text"] for line in analysis["recognitionResults"][0]["lines"]]
    #conf_infos = [word["text"] for word in analysis["recognitionResults"][0]["lines"]["words"]]
    #print(' :: ', line_infos)
    #print(' c :: ',analysis["recognitionResults"][0]["lines"])
    TotalProb=0
    for wb in analysis["recognitionResults"][0]["lines"]:
        sumProb=0
        sumLen=0
        for line in wb['words']:
            dummy=line['text']
            dotloc=dummy.find('.')
            if(dotloc<0):
                text=dummy
                text=re.sub("[^0-9]", "", dummy)
            else:
                text=''
            try:
                conf=line['confidence']
            except:
                conf='High'
            #print(' L! : ',text, ' ==> ',conf)
            length=len(text)
            if(conf=='Low'):
                Prob=0.5*length
            else:
                Prob=1.0*length
            sumLen=sumLen+length
            sumProb=sumProb+Prob

            #print(' wb : ', text, ' :: ', conf, ' :: ', sumLen, '  ::  ',sumProb)
        #print(' ==> ', sumLen, ' == ', sumProb)
        TotalProb=0.0
        if(sumLen<=6):
            TotalProb=sumProb/6.0
        else:
            TotalProb=sumProb/sumLen    
    #print(' total prob : ', TotalProb)



    word_infos = []
    for line in line_infos:
        dummy=line.strip().replace(" ","")
        word_infos.append(dummy)

    #print(' ==> ',word_infos)

    output=''
    for cha in word_infos:
        dotloc=cha.find('.')
        Xloc=cha.find('x')

        if(dotloc>0 or Xloc>=0):
            cha=''
        output=output+cha

    result1=re.sub("[^0-9]", "", output)

    if(len(result1)>6):
        result=result1[:6]
    else:
        result=result1

    #print(' result : ', result)
    #print(' nx2, ny2 : ',nx2,' :: ',ny2)

    polygons = []
    if ("recognitionResults" in analysis):
        # Extract the recognized text, with bounding boxes.
        polygons = [(line["boundingBox"], line["text"])
                    for line in analysis["recognitionResults"][0]["lines"]]


    # Display the image and overlay it with the extracted text.
    plt.figure(figsize=(15, 15))
    image=imgin
    #image = Image.open(image_path)
    
    ax = plt.imshow(image)
    for polygon in polygons:
        vertices = [(polygon[0][i], polygon[0][i+1])
                    for i in range(0, len(polygon[0]), 2)]
        text = polygon[1]
        patch = Polygon(vertices, closed=True, fill=False, linewidth=2, color='y')
        ax.axes.add_patch(patch)
        plt.text(vertices[0][0], vertices[0][1], text, fontsize=20, va="top")

    #plt.show()
    plt.close()
    return result, TotalProb

def Number_Reader_ReadAPI_3_1(image, ocr_url, subscription):

    nx, ny = image.size
    im2 = image.resize((int(nx*1), int(ny*1)), Image.BICUBIC)

    nx2, ny2 = im2.size
    
    img = np.array(im2)
    #dst = cv2.cvtColor(np.array(im2), cv2.COLOR_BGR2GRAY)
    #dst2 = cv2.cvtColor(np.array(im2), cv2.COLOR_BGR2GRAY)
    #img = cv2.fastNlMeansDenoising(dst2,None,6,7,21)
    #dst = cv2.fastNlMeansDenoising(np.array(im2),None,3,7,21)
    #dst2 = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    #(thresh, dst2) = cv2.threshold(dst, 105, 255, cv2.THRESH_BINARY)

    
    medfilter=median_filter(np.array(im2),1)
    lap=cv2.Laplacian(medfilter,cv2.CV_64F)
    img1=medfilter-0.7*lap
    dst = cv2.fastNlMeansDenoising(img1,None,6,7,21)
    img = cv2.Canny(dst,100,200)

    #img = cv2.medianBlur(np.array(im2),5)
    #img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)


    im_resize=img
    is_success, im_buf_arr = cv2.imencode(".jpg", im_resize)
    byte_im = im_buf_arr.tobytes()
   
    image_data=byte_im

    headers = {'Ocp-Apim-Subscription-Key': subscription, 'Content-Type': 'application/octet-stream'}
    #params = {'language': 'unk', 'detectOrientation': 'true'}
    params = {'language': 'unk'}

    response = requests.post(ocr_url, headers=headers, params=params, data = image_data)

    response.raise_for_status()

    #==============================

    # Holds the URI used to retrieve the recognized text.
    operation_url = response.headers["Operation-Location"]

    # The recognized text isn't immediately available, so poll to wait for completion.
    analysis = {}
    poll = True
    while (poll):
        response_final = requests.get(
            response.headers["Operation-Location"], headers=headers)
        analysis = response_final.json()
        #print(analysis)
        time.sleep(1)
        if ("recognitionResults" in analysis):
            poll = False
        if ("status" in analysis and analysis['status'] == 'Failed'):
            poll = False

    # Extract the word bounding boxes and text.
    line_infos = [line["text"] for line in analysis["recognitionResults"][0]["lines"]]
    #conf_infos = [word["text"] for word in analysis["recognitionResults"][0]["lines"]["words"]]
    #print(' :: ', line_infos)
    #print(' c :: ',analysis["recognitionResults"][0]["lines"])
    TotalProb=0
    for wb in analysis["recognitionResults"][0]["lines"]:
        sumProb=0
        sumLen=0
        for line in wb['words']:
            dummy=line['text']
            text=re.sub("[^0-9]", "", dummy)
            try:
                conf=line['confidence']
            except:
                conf='High'
            #print(' L! : ',text, ' ==> ',conf)
            length=len(text)
            if(conf=='Low'):
                Prob=0.5*length
            else:
                Prob=1.0*length
            sumLen=sumLen+length
            sumProb=sumProb+Prob

            #print(' wb : ', text, ' :: ', conf, ' :: ', sumLen, '  ::  ',sumProb)
        #print(' ==> ', sumLen, ' == ', sumProb)
        TotalProb=0.0
        if(sumLen<=6):
            TotalProb=sumProb/6.0
        else:
            TotalProb=sumProb/sumLen    
    #print(' total prob : ', TotalProb)



    word_infos = []
    for line in line_infos:
        dummy=line.strip().replace(" ","")
        word_infos.append(dummy)

    #print(' ==> ',word_infos)

    output=''
    for cha in word_infos:
        output=output+cha

    result1=re.sub("[^0-9]", "", output)
    if(len(result1)>6):
        result=result1[:6]
    else:
        result=result1

    #print(' result : ', result)
    #print(' nx2, ny2 : ',nx2,' :: ',ny2)

    polygons = []
    if ("recognitionResults" in analysis):
        # Extract the recognized text, with bounding boxes.
        polygons = [(line["boundingBox"], line["text"])
                    for line in analysis["recognitionResults"][0]["lines"]]


    # Display the image and overlay it with the extracted text.
    plt.figure(figsize=(15, 15))
    image=img
    #image = Image.open(image_path)
    
    ax = plt.imshow(image)
    for polygon in polygons:
        vertices = [(polygon[0][i], polygon[0][i+1])
                    for i in range(0, len(polygon[0]), 2)]
        text = polygon[1]
        patch = Polygon(vertices, closed=True, fill=False, linewidth=2, color='y')
        ax.axes.add_patch(patch)
        plt.text(vertices[0][0], vertices[0][1], text, fontsize=20, va="top")

    #plt.show()
    plt.close()
    return result, TotalProb

def Detect_Meter(img_path,ocr_url,headers):
    image_data_1 = open(img_path, "rb").read()
    img = cv2.imread(img_path)
    #dst = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
    
    dst2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dst = cv2.fastNlMeansDenoising(dst2,None,6,7,21)
    
    #dst = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dst=img

    success, encoded_image = cv2.imencode('.jpg', dst)
    image_data = encoded_image.tobytes()

    #print(' image_data : ',type(image_data))

    response = requests.post(ocr_url, headers=headers, data = image_data)
    response.raise_for_status()
    analysis = response.json()

    #print(' ==> analysis : ',analysis)

    # Extract the word bounding boxes and text.
    line_infos = analysis["predictions"]
    word_infos = []


    image1 = Image.open(img_path)

    #rotated = image.rotate(0)
    #try:
    #    # Grab orientation value.
    #    image_exif = image._getexif()
    #    image_orientation = image_exif[274]
        # Rotate depending on orientation.
    #    if image_orientation == 3:
    #        rotated = image.rotate(180)
    #    if image_orientation == 6:
    #        rotated = image.rotate(-90)
    #    if image_orientation == 8:
    #        rotated = image.rotate(90)
    # Save rotated image.
    #    rotated.save('rotated.jpg')
    #except:
    #    pass
    #image=rotated
    image = Image.fromarray( img, mode='RGB' )
    #image=cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

    im_width, im_height = image.size
    #im_height, im_width, channels = image.shape
    #print(' image : ',type(image),' :: ',type(image1), ' ==> ',im_width,':',im_height)
    ax = plt.imshow(image, alpha=0.5)

    df_line = pd.DataFrame(columns = ['prob', 'left','top','width','height'])

    #print(" - Probability - TagName - Left Loc. - Height Loc. -")
    for line in line_infos:
        if( line['probability']>0.1 ):  
            #print(' ==> ',line['probability'], '====> ',line['tagName'], ' ====> ',line['boundingBox']['left'], ' :: ',line['boundingBox']['height'])
            left=line['boundingBox']['left']
            top=line['boundingBox']['top']
            width=line['boundingBox']['width']
            height=line['boundingBox']['height']
            newrow= {'prob':line['probability'], 'left':left*im_width, 'top':top*im_height, 'width':width*im_width, 'height':height*im_height  }
            df_line = df_line.append(newrow, ignore_index=True)
            prob_text='{0:.1f}'.format(line['probability']*100)
            text=line['tagName']+', '+str(prob_text)+'%'
            origin = (left*im_width, top*im_height)
            patch = Rectangle(origin, width*im_width, height*im_height,fill=False, linewidth=2, color='y')
            ax.axes.add_patch(patch)
            plt.text(origin[0], origin[1], text, fontsize=8, weight="bold", va="top")

    df_line=df_line.sort_values(by=['prob'],ascending=False).reset_index()

    df_out=df_line.loc[df_line.prob==df_line.prob.max()]

    maxprob=df_out.prob.max()
    x=df_out['left'][0]
    w=df_out['width'][0]
    top=df_out['top'][0]
    h=df_out['height'][0]
    #print(' ==> ',x,w,top,h)
    cropped_img = img[int(top-0.5*h):int(top+1.5*h), int(x-0.5*w):int(x+1.5*w)]
    #image_path1 =r'C:\Users\70018928\Documents\Project 2019\Ad-hoc\Odometer_Image\Meter_Detection\Original_Image\pil_1.jpg'
    #image_path2 =r'C:\Users\70018928\Documents\Project 2019\Ad-hoc\Odometer_Image\Meter_Detection\Original_Image\cv2_1.jpg'
    #cv2.imwrite(image_path2,cropped_img)
   
    # cropped for calling api
    area=(x-0.5*w,top-0.1*h,x+1.5*w,top+2.0*h)
    cropped_img=image.crop(area)

    # cropped for masking
    area_m=(x-0.1*w,top-0.1*h,x+1.1*w,top+1.1*h)
    cropped_img_m=image.crop(area_m)

    # cropped for digit detection
    area1=(x-0*w,top-0*h,x+1*w,top+1*h)
    cropped_img1=image.crop(area1)


    #cropped_img=image[int(top-0.3*h):int(top+1.3*h), int(x-0.3*w):int(x+1.3*w)]
    #cv2.imshow('image',cropped_img)
    cv_area=[int(top-0.3*h),int(top+1.3*h), int(x-0.3*w),int(x+1.3*w)]
    #print(' ==>',type(cropped_img))
    #cropped_img.save(image_path1, quality=100)
 
    #print(' -- ',type(cropped_img))

    plt.axis("off")
    #plt.show()
    plt.close()
    return cropped_img, cropped_img_m, cropped_img1, cv_area, maxprob

def Number_Detection(img1,ocr_url,headers):
    image_data = img1
    #image_path1 =r'C:\Users\70018928\Documents\Project 2019\Ad-hoc\Odometer_Image\Meter_Detection\Original_Image\pil_1.jpg'
    #image_data.save(image_path1, quality=100)

    #img = cv2.imread(image_path)

    img = cv2.cvtColor(np.array(image_data), cv2.COLOR_RGB2BGR)

    #print(' ==== >>>>> ',type(img))
    #dst = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
    dst3 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dst2 = cv2.fastNlMeansDenoising(dst3,None,6,7,21)

    medfilter=median_filter(dst2,1)
    lap=cv2.Laplacian(medfilter,cv2.CV_64F)
    dst=medfilter-0.7*lap

    success, encoded_image = cv2.imencode('.jpg', dst)
    image_data = encoded_image.tobytes()

    #print(' image_data : ',type(image_data))
    response     = requests.post(ocr_url, headers=headers, data = image_data)
    response.raise_for_status()
    analysis = response.json()
    #print(' ==> analysis : ',analysis)
    # Extract the word bounding boxes and text.
    line_infos = analysis["predictions"]
    word_infos = []
    #image = Image.open(image_path)
    image=img1
    im_width, im_height = image.size
    #print(' image : ',type(image), ' ==> ',im_width,':',im_height)
    ax = plt.imshow(image, alpha=0.5)

    df_line = pd.DataFrame(columns = ['tag','prob','left','top','width','height'])

    print(" - Probability - TagName - Left Loc. - Top Loc. - Width  - Height  -")
    for line in line_infos:
        if( line['probability']>0.3 ):  
            left=line['boundingBox']['left']*im_width
            top=line['boundingBox']['top']*im_height
            width=line['boundingBox']['width']*im_width
            height=line['boundingBox']['height']*im_height
            newrow= {'tag':line['tagName'],'prob':line['probability'], 'left':left, 'top':top, 'width':width, 'height':height  }
            df_line = df_line.append(newrow, ignore_index=True)
            prob_text='{0:.1f}'.format(line['probability']*100)
            #print(' ==> ',line['probability'], '====> ',line['tagName'], ' ====> ',left, ' :: ',top, ' :: ',width, ' :: ',height)
            text=line['tagName']+', '+str(prob_text)+'%'
            origin = (left, top)
            patch = Rectangle(origin, width, height,fill=False, linewidth=2, color='y')
            ax.axes.add_patch(patch)
            plt.text(origin[0], origin[1], text, fontsize=8, weight="bold", va="top")

    df_line=df_line.sort_values(by=['left'],ascending=True).reset_index()
    df_line=df_line.drop(columns=['index']).reset_index()
    df_line['Right']=df_line['left']+df_line['width']
    #print(' df_line == > ',df_line)


    dummy=''
    dummy1=1
    for ch in df_line['tag']:
        dummy=dummy+ch
    for numb in df_line['prob']:
        dummy1=dummy1*numb
    Output=dummy
    Prob=dummy1
    
    plt.axis("off")
    #plt.show()
    plt.close()

    return Output, Prob

def Digit_Detection(img1,ocr_url,headers):
    image = img1
    nx, ny = image.size
    im2 = image.resize((int(nx*1), int(ny*1)), Image.BICUBIC)
    
    #im2=img1

    #image_path1 =r'C:\Users\70018928\Documents\Project 2019\Ad-hoc\Odometer_Image\Meter_Detection\Original_Image\pil_1.jpg'
    #image_data.save(image_path1, quality=100)

    #img = cv2.imread(image_path)

    img = cv2.cvtColor(np.array(im2), cv2.COLOR_RGB2BGR)

    #print(' ==== >>>>> ',type(img))
    #dst = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
    #dst3 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #dst2 = cv2.fastNlMeansDenoising(dst3,None,6,7,21)
    dst2 = cv2.fastNlMeansDenoising(img,None,7,7,21)

    medfilter=median_filter(dst2,1)
    lap=cv2.Laplacian(medfilter,cv2.CV_64F)
    dst=medfilter-0.7*lap

    success, encoded_image = cv2.imencode('.jpg', dst)
    image_data = encoded_image.tobytes()

    #print(' image_data : ',type(image_data))
    response     = requests.post(ocr_url, headers=headers, data = image_data)
    response.raise_for_status()
    analysis = response.json()
    #print(' ==> analysis : ',analysis)
    # Extract the word bounding boxes and text.
    line_infos = analysis["predictions"]
    word_infos = []
    #image = Image.open(image_path)
    image=img1
    im_width, im_height = image.size
    #print(' image : ',type(image), ' ==> ',im_width,':',im_height)
    ax = plt.imshow(image, alpha=0.5)

    df_line = pd.DataFrame(columns = ['tag','prob','left','top','width','height'])

    #print(" - Probability - TagName - Left Loc. - Top Loc. - Width  - Height  -")
    for line in line_infos:
        if( line['probability']>0.33 ):  
            left=line['boundingBox']['left']*im_width
            top=line['boundingBox']['top']*im_height
            width=line['boundingBox']['width']*im_width
            height=line['boundingBox']['height']*im_height
            newrow= {'tag':line['tagName'],'prob':line['probability'], 'left':left, 'top':top, 'width':width, 'height':height  }
            df_line = df_line.append(newrow, ignore_index=True)
            prob_text='{0:.1f}'.format(line['probability']*100)
            #print(' ==> ',line['probability'], '====> ',line['tagName'], ' ====> ',left, ' :: ',top, ' :: ',width, ' :: ',height)
            text=line['tagName']+', '+str(prob_text)+'%'
            origin = (left, top)
            patch = Rectangle(origin, width, height,fill=False, linewidth=2, color='y')
            ax.axes.add_patch(patch)
            plt.text(origin[0], origin[1], text, fontsize=8, weight="bold", va="top")

    df_line=df_line.sort_values(by=['left'],ascending=True).reset_index()
    df_line=df_line.drop(columns=['index']).reset_index()
    df_line['Right']=df_line['left']+df_line['width']
    #print(' df_line == > ',df_line)


    #dummy=''
    #dummy1=1
    #for ch in df_line['tag']:
    #    dummy=dummy+ch
    #for numb in df_line['prob']:
    #    dummy1=dummy1*numb
    #Output=dummy
    #Prob=dummy1
    
    plt.axis("off")
    #plt.show()
    plt.close()

    return df_line

def Number_Reader_ReadAPI_4(image, ocr_url, subscription, df_in, mac_cl_ocr_url, mac_subscription):
    #print(' df_in : ',df_in.shape, ' ==> ',type(image))
    count=0
    Number=''
    Digit=[]
    DProb=[]
    Prob=1
    for n in range(df_in.shape[0]):
        x=df_in['left'][count]
        w=df_in['width'][count]
        top=df_in['top'][count]
        h=df_in['height'][count]
        count=count+1
        print(' ==> ',x,w,top,h)

        area=(x,top,x+w,top+h)
        img=image.crop(area)
        ax = plt.imshow(img, alpha=0.5)
        # try:
        Read_Number, TotalProb=Number_Reader_ReadAPI_3(img, read_ocr_url, subscription)

        if(Read_Number==None or Read_Number==''):
            Read_Number, TotalProb=Number_Reader_Classification(img,mac_cl_ocr_url,mac_headers,df_in)
            
        Number=str(Read_Number)
        Digit.append(Read_Number)
        Prob=Prob*TotalProb
        DProb.append(TotalProb)
            
        # except:
        Read_Number=''
                   
        print(' => ',count, ' :: ', Number, ' == ', Prob)

        plt.axis("off")
        #plt.show()

    return Digit, DProb

def Number_Reader_Classification(image, ocr_url, subscription, df_in):
    #print(' df_in : ',df_in.shape, ' ==> ',type(image))
    count=0
    Digit=[]
    DProb=[]
    Number=''
    Prob=1
    for n in range(df_in.shape[0]):
        x=df_in['left'][count]
        w=df_in['width'][count]
        top=df_in['top'][count]
        h=df_in['height'][count]
        count=count+1
        print(' ==> ',x,w,top,h)
        area=(x,top,x+w,top+h)
        img=image.crop(area)
        ax = plt.imshow(image, alpha=0.5)
        try:
            Read_Number, TotalProb=Number_Detection_Classification(img,mac_cl_ocr_url,mac_headers)
            Number=Number+str(Read_Number)
            Digit.append(Read_Number)
            Prob=Prob*TotalProb
            DProb.append(TotalProb)
        except:
            Read_Number=''
            Prob=0
            Digit.append(Read_Number)
            DProb.append(TotalProb)

        plt.axis("off")
        #plt.show()
        plt.close()

    return Digit, DProb

def Number_Detection_Classification(img1,ocr_url,headers):
    image_data = img1
    #image_path1 =r'C:\Users\70018928\Documents\Project 2019\Ad-hoc\Odometer_Image\Meter_Detection\Original_Image\pil_1.jpg'
    #image_data.save(image_path1, quality=100)

    #img = cv2.imread(image_path)

    img = cv2.cvtColor(np.array(image_data), cv2.COLOR_RGB2BGR)

    print(' ==== >>>>> ',type(img))
    #dst = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
    dst3 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dst2 = cv2.fastNlMeansDenoising(dst3,None,6,7,21)

    medfilter=median_filter(dst2,1)
    lap=cv2.Laplacian(medfilter,cv2.CV_64F)
    dst=medfilter-0.7*lap

    success, encoded_image = cv2.imencode('.jpg', dst)
    image_data = encoded_image.tobytes()

    print(' image_data : ',type(image_data))
    response     = requests.post(ocr_url, headers=headers, data = image_data)
    response.raise_for_status()
    analysis = response.json()
    print(' ==> analysis : ',analysis)
    # Extract the word bounding boxes and text.
    line_infos = analysis["predictions"]
    word_infos = []
    #image = Image.open(image_path)
    image=img1
    im_width, im_height = image.size
    print(' image : ',type(image), ' ==> ',im_width,':',im_height)
    #ax = plt.imshow(image, alpha=0.5)

    df_line = pd.DataFrame(columns = ['tag','prob'])

    print(" - Probability - TagName")
    for line in line_infos:
        if( line['probability']>0.2 ):  
            newrow= {'tag':line['tagName'],'prob':line['probability'] }
            df_line = df_line.append(newrow, ignore_index=True)
            prob_text='{0:.1f}'.format(line['probability']*100)
            print(' ==> ',line['probability'], '====> ',line['tagName'])
            text=line['tagName']+', '+str(prob_text)+'%'
            
    df_line=df_line.sort_values(by=['prob'],ascending=False).reset_index()
    df_out=df_line.loc[df_line.prob==df_line.prob.max()]

    maxprob=df_out.prob.max()
    #df_line=df_line.drop(columns=['index']).reset_index()
    #print(' max prob : ',maxprob)

    print(' df_line == > ',df_line['tag'].iloc[0], ' :: ',df_line['prob'].iloc[0])

    Output=df_line['tag'].iloc[0]
    Prob=df_line['prob'].iloc[0]

    return Output, Prob

def Order_Number(df_digit, Digit, DProb):
    D_zip = dict(zip(Digit, DProb))
    df_data=pd.DataFrame(columns = ['Number', 'Prob'])
    for n in Digit:
        newrow= {'Number':n, 'Prob':D_zip[n]}
        df_data = df_data.append(newrow, ignore_index=True)

    df_digit=pd.concat([df_digit, df_data],axis=1)            
    #print(' df : ', df_digit)
    listD=df_digit['Number'].values.tolist()
    listP=df_digit['Prob'].values.tolist()
    #print(' List : ', len(listD))
    dummy=''
    for n in listD:
        dummy=dummy+n
    dummyP=1
    for n in listP:
        dummyP=dummyP*n
    #print(' ==> Number : ', dummy, ', Prob : ',dummyP)
    return dummy, dummyP

def Detect_Meter_2(img,ocr_url,headers):
    #dst = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
    
    dst2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dst = cv2.fastNlMeansDenoising(dst2,None,6,7,21)
    
    #dst = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dst=img

    success, encoded_image = cv2.imencode('.jpg', dst)
    image_data = encoded_image.tobytes()

    print(' image_data : ',type(image_data))

    response = requests.post(ocr_url, headers=headers, data = image_data)
    response.raise_for_status()
    analysis = response.json()

    print(' ==> analysis : ',analysis)

    # Extract the word bounding boxes and text.
    line_infos = analysis["predictions"]
    word_infos = []


    

    #rotated = image.rotate(0)
    #try:
    #    # Grab orientation value.
    #    image_exif = image._getexif()
    #    image_orientation = image_exif[274]
        # Rotate depending on orientation.
    #    if image_orientation == 3:
    #        rotated = image.rotate(180)
    #    if image_orientation == 6:
    #        rotated = image.rotate(-90)
    #    if image_orientation == 8:
    #        rotated = image.rotate(90)
    # Save rotated image.
    #    rotated.save('rotated.jpg')
    #except:
    #    pass
    #image=rotated
    image = Image.fromarray( img, mode='RGB' )
    #image=cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

    im_width, im_height = image.size
    #im_height, im_width, channels = image.shape
    print(' image : ',type(image),' :: ',im_width,':',im_height)
    ax = plt.imshow(image, alpha=0.5)

    df_line = pd.DataFrame(columns = ['prob', 'left','top','width','height'])

    print(" - Probability - TagName - Left Loc. - Height Loc. -")
    for line in line_infos:
        if( line['probability']>0.1 ):  
            print(' ==> ',line['probability'], '====> ',line['tagName'], ' ====> ',line['boundingBox']['left'], ' :: ',line['boundingBox']['height'])
            left=line['boundingBox']['left']
            top=line['boundingBox']['top']
            width=line['boundingBox']['width']
            height=line['boundingBox']['height']
            newrow= {'prob':line['probability'], 'left':left*im_width, 'top':top*im_height, 'width':width*im_width, 'height':height*im_height  }
            df_line = df_line.append(newrow, ignore_index=True)
            prob_text='{0:.1f}'.format(line['probability']*100)
            text=line['tagName']+', '+str(prob_text)+'%'
            origin = (left*im_width, top*im_height)
            patch = Rectangle(origin, width*im_width, height*im_height,fill=False, linewidth=2, color='y')
            ax.axes.add_patch(patch)
            plt.text(origin[0], origin[1], text, fontsize=8, weight="bold", va="top")

    df_line=df_line.sort_values(by=['prob'],ascending=False).reset_index()

    df_out=df_line.loc[df_line.prob==df_line.prob.max()]

    maxprob=df_out.prob.max()
    x=df_out['left'][0]
    w=df_out['width'][0]
    top=df_out['top'][0]
    h=df_out['height'][0]
    #print(' ==> ',x,w,top,h)
    cropped_img = img[int(top-0.3*h):int(top+1.3*h), int(x-0.3*w):int(x+1.3*w)]
    #image_path1 =r'C:\Users\70018928\Documents\Project 2019\Ad-hoc\Odometer_Image\Meter_Detection\Original_Image\pil_1.jpg'
    #image_path2 =r'C:\Users\70018928\Documents\Project 2019\Ad-hoc\Odometer_Image\Meter_Detection\Original_Image\cv2_1.jpg'
    #cv2.imwrite(image_path2,cropped_img)
   
    area=(x-0.3*w,top-0.1*h,x+1.3*w,top+2.0*h)
    cropped_img=image.crop(area)

    # cropped for masking
    area_m=(x-0.1*w,top-0.1*h,x+1.1*w,top+1.1*h)
    cropped_img_m=image.crop(area_m)

    # cropped for digit detection
    area1=(x-0*w,top-0*h,x+1*w,top+1*h)
    cropped_img1=image.crop(area1)

    #cropped_img=image[int(top-0.3*h):int(top+1.3*h), int(x-0.3*w):int(x+1.3*w)]
    #cv2.imshow('image',cropped_img)
    cv_area=[int(top-0.3*h),int(top+1.3*h), int(x-0.3*w),int(x+1.3*w)]
    #print(' ==>',type(cropped_img))
    #cropped_img.save(image_path1, quality=100)
 
    #print(' -- ',type(cropped_img))

    plt.axis("off")
    #plt.show()
    plt.close()
    return cropped_img, cropped_img_m,cropped_img1, cv_area, maxprob
    
def Number_Reader_Classification_2(image, ocr_url, subscription, df_in, imgpath, series, number):
    #print(' df_in : ',df_in.shape, ' ==> ',type(image))
    count=0
    Digit=[]
    DProb=[]
    Number=''
    Prob=1
    
    for n in range(df_in.shape[0]):
        x=df_in['left'][count]
        w=df_in['width'][count]
        top=df_in['top'][count]
        h=df_in['height'][count]
        count=count+1
        #print(' ==> ',x,w,top,h)

        area=(x,top,x+w,top+h)
        img=image.crop(area)
        #ax = plt.imshow(image, alpha=0.5)
        filename=imgpath+series+'-'+str(number)+'-C'+str(count)+'.jpg'
        img.save(filename, quality=100)


        #plt.axis("off")
        #plt.show()

def Detect_Meter_3(img_path,ocr_url,headers):
    image_data_1 = open(img_path, "rb").read()
    img = cv2.imread(img_path)
    #dst = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
    
    dst2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dst = cv2.fastNlMeansDenoising(dst2,None,6,7,21)
    
    #dst = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dst=img

    success, encoded_image = cv2.imencode('.jpg', dst)
    image_data = encoded_image.tobytes()

    #print(' image_data : ',type(image_data))

    response = requests.post(ocr_url, headers=headers, data = image_data)
    response.raise_for_status()
    analysis = response.json()

    print(' ==> analysis : ',analysis)

    # Extract the word bounding boxes and text.
    line_infos = analysis["predictions"]
    word_infos = []


    image1 = Image.open(img_path)

    #rotated = image.rotate(0)
    #try:
    #    # Grab orientation value.
    #    image_exif = image._getexif()
    #    image_orientation = image_exif[274]
        # Rotate depending on orientation.
    #    if image_orientation == 3:
    #        rotated = image.rotate(180)
    #    if image_orientation == 6:
    #        rotated = image.rotate(-90)
    #    if image_orientation == 8:
    #        rotated = image.rotate(90)
    # Save rotated image.
    #    rotated.save('rotated.jpg')
    #except:
    #    pass
    #image=rotated
    image = Image.fromarray( img, mode='RGB' )
    #image=cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

    im_width, im_height = image.size
    #im_height, im_width, channels = image.shape
    print(' image : ',type(image),' :: ',type(image1), ' ==> ',im_width,':',im_height)
    ax = plt.imshow(image, alpha=0.5)

    df_line = pd.DataFrame(columns = ['prob', 'left','top','width','height'])

    print(" - Probability - TagName - Left Loc. - Height Loc. -")
    for line in line_infos:
        if( line['probability']>0.1 ):  
            print(' ==> ',line['probability'], '====> ',line['tagName'], ' ====> ',line['boundingBox']['left'], ' :: ',line['boundingBox']['height'])
            left=line['boundingBox']['left']
            top=line['boundingBox']['top']
            width=line['boundingBox']['width']
            height=line['boundingBox']['height']
            newrow= {'prob':line['probability'], 'left':left*im_width, 'top':top*im_height, 'width':width*im_width, 'height':height*im_height  }
            df_line = df_line.append(newrow, ignore_index=True)
            prob_text='{0:.1f}'.format(line['probability']*100)
            text=line['tagName']+', '+str(prob_text)+'%'
            origin = (left*im_width, top*im_height)
            patch = Rectangle(origin, width*im_width, height*im_height,fill=False, linewidth=2, color='y')
            ax.axes.add_patch(patch)
            plt.text(origin[0], origin[1], text, fontsize=8, weight="bold", va="top")

    df_line=df_line.sort_values(by=['prob'],ascending=False).reset_index()

    print(' ==> ', df_line)


    plt.axis("off")
    #plt.show()
    plt.close()

    print(' df_in : ',df_in.shape, ' ==> ',type(img))
    Digit=[]
    DProb=[]
    Number=''
    Prob=1
    TotalProb=0
    for n in range(df_in.shape[0]):
        x=df_in['left'][count]
        w=df_in['width'][count]
        top=df_in['top'][count]
        h=df_in['height'][count]
        count=count+1
        print(' ==> ',x,w,top,h)

        area=(x,top,x+w,top+h)
        img1=img.crop(area)
        ax = plt.imshow(img, alpha=0.5)
        try:
            Read_Number, TotalProb=Number_Detection_Classification(img1,mac_cl_ocr_url,mac_headers)
            Number=str(Read_Number)
            Digit.append(Read_Number)
            Prob=Prob*TotalProb
            DProb.append(TotalProb)
        except:
            Read_Number=''
            Prob=0
            Digit.append(Read_Number)
            DProb.append(TotalProb)
            

        plt.axis("off")
        #plt.show()
        plt.close()

    return Digit, DProb

def Number_Reader_ReadAPI_4_2(image, ocr_url, subscription, df_in, mac_ocr_url, mac_headers):
    #print(' df_in : ',df_in.shape, ' ==> ',type(image))
    count=0
    Number=''
    Digit=[]
    DProb=[]
    Prob=1
    for n in range(df_in.shape[0]):
        x=df_in['left'][count]
        w=df_in['width'][count]
        top=df_in['top'][count]
        h=df_in['height'][count]
        count=count+1
        
        #print(' ==> ',x,w,top,h)

        s=1.1
        area=x,top,x+(w*s),top+(h*s)
        img=image.crop(area)
        ax = plt.imshow(img, alpha=0.5)
        # try:
        Read_Number, TotalProb=Number_Reader_ReadAPI_3(img, read_ocr_url, subscription)

        if(Read_Number==None or Read_Number=='' or len(Read_Number)>1 ):
           # Read_Number, TotalProb=Number_Reader_Classification(img,mac_cl_ocr_url,mac_headers,df_in)
 
         try:
            #Read_Number, TotalProb=Number_Detection_Classification_2(img,mac_cl_ocr_url,mac_headers)
            Read_Number, TotalProb=Number_Detection_Classification_2(img,mac_GFS_cl_ocr_url,mac_headers)
            #Read_Number, TotalProb=Number_Reader_Classification(img,mac_cl_ocr_url,mac_headers,df_in)

         except:
            Read_Number=''
            print('ERROR JA')

        Number=str(Read_Number)
        Digit.append(Read_Number)
        Prob=Prob*TotalProb
        DProb.append(TotalProb)
                          
        print(' => ',count, ' :: ', Number, ' == ', Prob)

        plt.axis("off")
        #plt.show()
        
    return Digit, DProb

def Number_Detection_Classification_2(img,mac_ocr_url,mac_headers):
    image_data = img
    #image_path1 =r'C:\Users\70018928\Documents\Project 2019\Ad-hoc\Odometer_Image\Meter_Detection\Original_Image\pil_1.jpg'
    #image_data.save(image_path1, quality=100)
    #img = cv2.imread(image_path)
    img = cv2.cvtColor(np.array(image_data), cv2.COLOR_RGB2BGR)
    #print(' ==== >>>>> ',type(img))
    #print('RUN!!!==== >>>>Number_Detection_Classification_2')
    #dst = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
    dst3 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dst2 = cv2.fastNlMeansDenoising(dst3,None,6,7,21)
    #dst2 = cv2.fastNlMeansDenoising(img,None,7,7,21)
    medfilter=median_filter(dst2,1)
    lap=cv2.Laplacian(medfilter,cv2.CV_64F)
    dst=medfilter-0.7*lap

    success, encoded_image = cv2.imencode('.jpg', dst)
    image_data = encoded_image.tobytes()

    #print(' image_data : ',type(image_data))
    #plt.show(image_data)
    
    response = requests.post(mac_ocr_url, headers=mac_headers, data = image_data)
    response.raise_for_status()
    analysis = response.json()
    #print(' ==> analysis : ',analysis)
    # Extract the word bounding boxes and text.
    line_infos = analysis["predictions"]
    word_infos = []
    #image = Image.open(image_path)
    image=image_data
    #im_width, im_height = image.size
    #print(' image : ',type(image), ' ==> ',im_width,':',im_height)
    #ax = plt.imshow(image, alpha=0.5)

    df_line = pd.DataFrame(columns = ['tag','prob'])
    
    #print(" - Probability - TagName")
    for line in line_infos:
        if( line['probability']>0.3 ):  
            newrow= {'tag':line['tagName'],'prob':line['probability'] }
            df_line = df_line.append(newrow, ignore_index=True)
            prob_text='{0:.1f}'.format(line['probability']*100)
            #print(' ==> ',line['probability'], '====> ',line['tagName'])
            text=line['tagName']+', '+str(prob_text)+'%'
            
    df_line=df_line.sort_values(by=['prob'],ascending=False).reset_index()
    df_out=df_line.loc[df_line.prob==df_line.prob.max()]

    maxprob=df_out.prob.max()
    #df_line=df_line.drop(columns=['index']).reset_index()
    #print(' max prob : ',maxprob)

    #print(' df_line == > ',df_line['tag'].iloc[0][:1], ' :: ',df_line['prob'].iloc[0])
    Output=df_line['tag'].iloc[0][:1]
    Prob=df_line['prob'].iloc[0]

    return Output, Prob
