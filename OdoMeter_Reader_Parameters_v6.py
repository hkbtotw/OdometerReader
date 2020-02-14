import os
import sys
import requests
import time
import cv2
import glob
import re
import pandas as pd
import numpy as np
from io import BytesIO
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from PIL import Image,  ImageEnhance,ExifTags, ImageDraw
from scipy.ndimage.filters import median_filter

# Mac Endpoint (Use if TT enpoint is out of limit at certain period of time due to free plan selection)
mac_ep2="https://southeastasia.api.cognitive.microsoft.com/"
mac_sub2='2931247984204b1c997ad8ebcc1c80b7'

# Meter Detection
dm_endpoint="https://southeastasia.api.cognitive.microsoft.com/"
subscription='2931247984204b1c997ad8ebcc1c80b7'
dm_ocr_url = dm_endpoint + "customvision/v3.0/Prediction/95dba6d3-4d44-4b23-bad3-dd9d0a15090e/detect/iterations/Iteration3/image"


# Digit location
mac_endpoint="https://southeastasia.api.cognitive.microsoft.com/"
mac_ocr_url = mac_endpoint + "customvision/v3.0/Prediction/b62e724e-6590-4317-8424-32f96223f8dd/detect/iterations/Iteration35/image"

mac_subscription='2931247984204b1c997ad8ebcc1c80b7'
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

# Add size of image to be passed to Custom Vision API
# by applying the target image to black patch background
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

# Custom vision - Batch Read API with Image as input (outdated)
def Number_Reader_ReadAPI_3(image, ocr_url, subscription):

    nx, ny = image.size
    im2 = image.resize((int(nx*0.7), int(ny*0.7)), Image.BICUBIC)
    #im2 = image.resize((int(nx*1.0), int(ny*1.0)), Image.BICUBIC)

    nx2, ny2 = im2.size
    
    im_cv = np.array(im2)
    #img1 = cv2.cvtColor(im_cv, cv2.COLOR_BGR2RGB)
    img1 = cv2.cvtColor(np.array(im2), cv2.COLOR_BGR2GRAY)

    #dst2 = cv2.cvtColor(np.array(im2), cv2.COLOR_BGR2GRAY)
    img2 = cv2.fastNlMeansDenoising(img1,None,3,7,21)
    #plt.imshow(img)
    #plt.show()
    img = cv2.bilateralFilter(img2,9,9,9)
    #plt.imshow(img)
    #plt.show()
    #dst = cv2.fastNlMeansDenoising(np.array(im2),None,3,7,21)
    #dst2 = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    ret,thresh_tozero = cv2.threshold(img,125,255,cv2.THRESH_TOZERO)
    #thresh_mean = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    #(thresh, dst2) = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    #im,contours,hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 
    #with_contours = cv2.drawContours(img,contours,-1,(0,255,0),3) 
    #dst2 = cv2.Canny(img,100,200) 
    #dst2 = cv2.fastNlMeansDenoising(dst,None,6,7,21)
    medfilter=median_filter(thresh_tozero,1)
    lap=cv2.Laplacian(medfilter,cv2.CV_64F)
    img3=medfilter-0.01*lap

    #img3 = cv2.medianBlur(img2,5)
    #img = cv2.adaptiveThreshold(img3,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    #print(' img type: ',type(img), ' ==> ',img.shape)
    imgin=rescale_image(img3)
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

# Custom vision - Batch Read API with filepath as input (outdated)
def Number_Reader_ReadAPI_3_1(img_path, ocr_url, subscription):
    #image_data_1 = open(img_path, "rb").read()
    img1 = cv2.imread(img_path)
    #nx, ny = img1.size
    #im2 = img1.resize((int(nx*1), int(ny*1)), Image.BICUBIC)
    #nx2, ny2 = im2.size
    
    #img = np.array(img1)
    #dst = cv2.cvtColor(np.array(im2), cv2.COLOR_BGR2GRAY)
    #dst2 = cv2.cvtColor(np.array(im2), cv2.COLOR_BGR2GRAY)
    #img = cv2.fastNlMeansDenoising(dst2,None,6,7,21)
    #dst = cv2.fastNlMeansDenoising(np.array(im2),None,3,7,21)
    #dst2 = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    #(thresh, dst2) = cv2.threshold(dst, 105, 255, cv2.THRESH_BINARY)

    
    #medfilter=median_filter(np.array(img1),1)
    #lap=cv2.Laplacian(medfilter,cv2.CV_64F)
    #img2=medfilter-0.7*lap
    #dst = cv2.fastNlMeansDenoising(img2,None,6,7,21)
    #img = cv2.Canny(dst,100,200)

    #img = cv2.medianBlur(np.array(im2),5)
    #img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)


    #im_resize=img
    is_success, im_buf_arr = cv2.imencode(".jpg", img1)
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
    image=img1
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

# Custom vision - Object Detection ( Detect DigitBar )  (Active)
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
            newrow= {'tag':line['tagName'],'prob':line['probability'], 'left':left*im_width, 'top':top*im_height, 'width':width*im_width, 'height':height*im_height  }
            df_line = df_line.append(newrow, ignore_index=True)
            prob_text='{0:.1f}'.format(line['probability']*100)
            text=line['tagName']+', '+str(prob_text)+'%'
            #print(' tagname : ', text)
            origin = (left*im_width, top*im_height)
            patch = Rectangle(origin, width*im_width, height*im_height,fill=False, linewidth=2, color='y')
            ax.axes.add_patch(patch)
            plt.text(origin[0], origin[1], text, fontsize=8, weight="bold", va="top")

    df_line=df_line.sort_values(by=['prob'],ascending=False).reset_index()
    #print(' df_line : ',df_line)

    df_out=df_line.loc[df_line.prob==df_line.prob.max()]
    

    maxprob=df_out.prob.max()
    meterType=df_out['tag'][0]
    #print(' ==> ', meter_type)
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
    area=(x-0.1-0.5*w,top-0.1*h,x+1.5*w,top+2.0*h)
    cropped_img=image.crop(area)

    # cropped for masking
    area_m=(x-0.1*w,top-0.1*h,x+1.1*w,top+1.05*h)
    cropped_img_m=image.crop(area_m)

    # cropped for digit detection
    area1=(x-0.1*w,top-0.1*h,x+1.1*w,top+1.1*h)  # Acc 87%
    #area1=(x+0.05*w,top-0.1*h,x+1.1*w,top+1.1*h)
    #area1=(x-0*w,top-0*h,x+1*w,top+1*h)   #  Acc.62%
    #area1=(x-0.05*w,top-0.1*h,x+1.1*w,top+1.1*h)   #  Acc.62%
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
    return cropped_img, cropped_img_m, cropped_img1, cv_area, maxprob, meterType

# Custom vision - Batch Read API with Image as input - GrayScale Filter for Analog & Digital (Active)
def Number_Reader_ReadAPI_3_4(image, ocr_url, subscription, meterType):

    nx, ny = image.size
    im2 = image.resize((int(nx*0.7), int(ny*0.7)), Image.BICUBIC)
    #im2 = image.resize((int(nx*1.0), int(ny*1.0)), Image.BICUBIC)

    nx2, ny2 = im2.size
    
    im_cv = np.array(im2)
    #img1 = cv2.cvtColor(im_cv, cv2.COLOR_BGR2RGB)
    img1 = cv2.cvtColor(np.array(im2), cv2.COLOR_BGR2GRAY)

    #dst2 = cv2.cvtColor(np.array(im2), cv2.COLOR_BGR2GRAY)
    img2 = cv2.fastNlMeansDenoising(img1,None,3,7,21)
    #plt.imshow(img)
    #plt.show()
    img = cv2.bilateralFilter(img2,9,9,9)
    #plt.imshow(img)
    #plt.show()
    #dst = cv2.fastNlMeansDenoising(np.array(im2),None,3,7,21)
    #dst2 = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    #ret,thresh_tozero = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
    #(thresh, dst2) = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    #im,contours,hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 
    #with_contours = cv2.drawContours(img,contours,-1,(0,255,0),3) 
    #dst2 = cv2.Canny(img,100,200) 
    #dst2 = cv2.fastNlMeansDenoising(dst,None,6,7,21)
    medfilter=median_filter(img,1)
    lap=cv2.Laplacian(medfilter,cv2.CV_64F)
    img3=medfilter-0.01*lap

    #img3 = cv2.medianBlur(img2,5)
    #img = cv2.adaptiveThreshold(img3,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    #print(' img type: ',type(img), ' ==> ',img.shape)
    imgin=rescale_image(img3)
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

    #################################################
    # Original Output
    #output=''
    #for cha in word_infos:
    #    dotloc=cha.find('.')
    #    Xloc=cha.find('x')
    #    if(dotloc>0 or Xloc>=0):
    #        cha=''
    #    output=output+cha
    #result1=re.sub("[^0-9]", "", output)
    #if(len(result1)>6):
    #    result=result1[:6]
    #else:
    #    result=result1
    #####################################################

    #print(' result : ', result)
    #print(' nx2, ny2 : ',nx2,' :: ',ny2)

    polygons = []
    if ("recognitionResults" in analysis):
        # Extract the recognized text, with bounding boxes.
        polygons = [(line["boundingBox"], line["text"])
                    for line in analysis["recognitionResults"][0]["lines"]]

    ###############################
    # New Text output by selecting the prediction returned from Batch Read API with widest box     
    polyLen=len(polygons)
    print(' pL : ',polyLen)
    polyMax=0
    nMax=0
    nPos=[]
    nText=''
    for n in range(polyLen):
        #print(' :: ', n)
        polyDiff=polygons[n][0][2]-polygons[n][0][0]
        if(polyDiff>polyMax):
            nMax=n
            polyMax=polyDiff
            nPos=polygons[n][0]
            nText=polygons[n][1]
            print(' ==> ', nMax, ' --- ',nPos, ' ==== ',nText )
    print(' F ==> ', nMax, ' --- ',nPos, ' ==== ',nText )
    result1=re.sub("[^0-9]", "", nText)
    rlen=len(result1)-6
    print(' rlen : ', rlen, ' ==> ', len(result1))
    
    
    def ReversedString(Sin):
        reversedString=[]
        index = len(Sin) # calculate length of string and save in index
        while index > 0: 
            reversedString += Sin[ index - 1 ] # save the value of str[index-1] in reverseString
            index = index - 1 # decrement index
        print(reversedString) # reversed string
        RS=''
        for ir in reversedString:
            RS=RS+ir
        return RS

    if(meterType=="d_meter"):
        # Code for Digital Meter , Need reversed check
        result2=[]
        constNum=6   #for digital
        if(len(result1)>constNum):
            count=0
            for ir in range(len(result1),-1,-1):
                count+=1
                #print(ir)
                result2.append(result1[ir-1])
                if(count==constNum):
                    break
            result=ReversedString(result2)
            print( ' ******** d_meter ******** ')
        else:
            result=result1
            print( ' ******** a_meter ******** ')

    else:
        # Code for Analog Meter, 
        rlen=len(result1)
        if(rlen>=6):
            result= result1[0:6]
        else:
            result=result1
    ######################################################



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

# Custom vision - Batch Read API with Image as input - Binary Filter for Digital (Active)
def Number_Reader_ReadAPI_3_5(image, ocr_url, subscription,meterType):


    """
    #image.show()
    img_origin=image
    nx, ny = image.size
    #im2 = image.resize((int(nx*1.0), int(ny*1.0)), Image.BICUBIC)
    im2 = image.resize((int(nx*1.0), int(ny*1.0)), Image.BICUBIC)
    nx2, ny2 = im2.size
    
    im_cv = np.array(im2)
    img3 = cv2.cvtColor(im_cv, cv2.COLOR_BGR2RGB)
    img1 = cv2.cvtColor(np.array(im2), cv2.COLOR_BGR2GRAY)

    dst2 = cv2.cvtColor(np.array(im2), cv2.COLOR_BGR2GRAY)
    img2 = cv2.fastNlMeansDenoising(img1,None,6,21,21)

    (thresh, dst) = cv2.threshold(dst2, 180, 255, cv2.THRESH_BINARY)

    
    blurred = cv2.GaussianBlur(im_cv, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 200, 255)
    #plt.imshow(edges)
    #plt.show()

    
    #img = cv2.bilateralFilter(dst,9,9,9)
    
    # Write out processed image for checking 
    #image_path_output =r'C:/Users/70018928/Documents/Project2020/TruckOdometer/20200203/Test_SSM_1/out_image/'
    #filename=image_path_output+'-'+str(fcount)+'-'+str(count)+'.jpg'
    #cv2.imwrite(filename, img2) 

    
    #plt.imshow(img)
    #plt.show()
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

    #imgin=rescale_image(img3)
    imgin=rescale_image(img2)
    #imgin=dst
    #plt.imshow(imgin)
    #plt.show()
    """
    nx, ny = image.size
    im2 = image.resize((int(nx*0.7), int(ny*0.7)), Image.BICUBIC)
    #im2 = image.resize((int(nx*1.0), int(ny*1.0)), Image.BICUBIC)

    nx2, ny2 = im2.size
    
    im_cv = np.array(im2)
    #img1 = cv2.cvtColor(im_cv, cv2.COLOR_BGR2RGB)
    img1 = cv2.cvtColor(np.array(im2), cv2.COLOR_BGR2GRAY)

    #dst2 = cv2.cvtColor(np.array(im2), cv2.COLOR_BGR2GRAY)
    img2 = cv2.fastNlMeansDenoising(img1,None,3,7,21)
    #plt.imshow(img)
    #plt.show()
    img = cv2.bilateralFilter(img2,9,9,9)
    #plt.imshow(img)
    #plt.show()
    #dst = cv2.fastNlMeansDenoising(np.array(im2),None,3,7,21)
    #dst2 = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    ret,thresh_tozero = cv2.threshold(img,125,255,cv2.THRESH_TOZERO)
    #thresh_mean = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    #(thresh, dst2) = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    #im,contours,hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 
    #with_contours = cv2.drawContours(img,contours,-1,(0,255,0),3) 
    #dst2 = cv2.Canny(img,100,200) 
    #dst2 = cv2.fastNlMeansDenoising(dst,None,6,7,21)
    medfilter=median_filter(thresh_tozero,1)
    lap=cv2.Laplacian(medfilter,cv2.CV_64F)
    img3=medfilter-0.01*lap

    #img3 = cv2.medianBlur(img2,5)
    #img = cv2.adaptiveThreshold(img3,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    #print(' img type: ',type(img), ' ==> ',img.shape)
    imgin=rescale_image(img3)



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
    print(' c :: ',analysis["recognitionResults"][0]["lines"])
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

    #######################################
    ## Original Outputting Text 
    #output=''
    #for cha in word_infos:
    #    dotloc=cha.find('.')
    #    Xloc=cha.find('x')
    #    if(dotloc>0 or Xloc>=0):
    #        cha=''
    #    output=output+cha
    #result1=re.sub("[^0-9]", "", output)
    #result1=output
    #if(len(result1)>6):
    #    result=result1[:6]
    #else:
    #    result=result1
    ####################################

    #print(' result : ', result)
    #print(' nx2, ny2 : ',nx2,' :: ',ny2)

    polygons = []
    if ("recognitionResults" in analysis):
        # Extract the recognized text, with bounding boxes.
        polygons = [(line["boundingBox"], line["text"])
                    for line in analysis["recognitionResults"][0]["lines"]]
        print(' PG : ',polygons, ' == ', type(polygons), ' :: ',len(polygons))

    # New Text output by selecting the prediction returned from Batch Read API with widest box     
    polyLen=len(polygons)
    print(' pL : ',polyLen)
    polyMax=0
    nMax=0
    nPos=[]
    nText=''
    for n in range(polyLen):
        #print(' :: ', n)
        polyDiff=polygons[n][0][2]-polygons[n][0][0]
        if(polyDiff>polyMax):
            nMax=n
            polyMax=polyDiff
            nPos=polygons[n][0]
            nText=polygons[n][1]
            print(' ==> ', nMax, ' --- ',nPos, ' ==== ',nText )
       
    print(' F ==> ', nMax, ' --- ',nPos, ' ==== ',nText )
    result=re.sub("[^0-9]", "", nText)

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

    ### Use this code below to crop only the widest predicted box returned from Batch Read API
    #i_width, i_height = Image.fromarray(imgin).size
    #print(' i : ',i_width, '  :: ',i_height)
    #x=nPos[0]
    #w=nPos[2]-nPos[0]
    #h=nPos[5]-nPos[1]
    #top=nPos[1]

    #print(' x :',x,' : ',top,' : ',w,' : ',h)
    #area=(x-0.*w,top-0.*h,x+1.*w,top+1.*h)
    #img_out=Image.fromarray(imgin).crop(area)

    #image_path_output =r'C:/Users/70018928/Documents/Project2020/TruckOdometer/20200203/Test_SSM_1/out_image/'
    #image_path1=image_path_output+'-1-'+str(fcount)+'-'+str(count)+'.jpg'
    #img_out.save(image_path1, quality=100)


    return result, TotalProb