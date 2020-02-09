import os
import glob
import numpy as np
import pandas as pd
import re
import OdoMeter_Reader_Parameters_v4 as orp1
from OdoMeter_Reader_Parameters_v4 import Digit_Detection as dd
from OdoMeter_Reader_Parameters_v4 import Number_Reader_ReadAPI_4_2
#from OdoMeter_Reader_Parameters_v4 import Number_Reader_Classification
from OdoMeter_Reader_Parameters_v4 import Order_Number
from OdoMeter_Reader_Parameters_v4 import Detect_Meter as dm
from OdoMeter_Reader_Parameters_v4 import Detect_Meter_2 as dm2
from OdoMeter_Reader_Parameters_v4 import Number_Reader_ReadAPI_3 as readapi_3
from OdoMeter_Reader_Parameters_v4 import Number_Reader_ReadAPI_3_1 as readapi_3_1
from OdoMeter_Reader_Parameters_v4 import IfLicensePlatTag
from OdoMeter_Reader_Parameters_v4 import MaskArea
from OdoMeter_Reader_Parameters_v4 import MaskArea_2
from OdoMeter_Reader_Parameters_v4 import Number_Detection_ImageProc as NDM
from PIL import Image,  ImageEnhance
import cv2

# Get file name and location
# Specify directory in which the images are kept
image_path="C:/Users/70018928/Documents/GitHub/OdometerReader/ImageData/"

image_path_output="C:/Users/70018928/Documents/GitHub/OdometerReader/Output_TestImage/"

path=image_path+"*.jpg"
files = []
for file in glob.glob(path):
    files.append(file)
print(' ==> ',files)

#Prepare table
df_Data=pd.DataFrame(columns = ['filename','Meter(Confidence)', 'ReadAPI(ALL)', 'ReadAPI(Separated)','Classification'])

#================================================================================
#Specify input path
count = 0
for n in files:
    head, tail = os.path.split(n)
    filename = tail[:len(tail)-4]
    #print(' ==> ',filename )

    count+=1
    TotalProb=0
    Read_Number=""
    strnum=""
    classifiedStrNum=""
    # Detect Digitbar on the meter (Analog or Digital meter) with Custom vision - Object detection
    try:
        

        Read_Number, TotalProb=readapi_3_1(n, orp1.read_ocr_url, orp1.subscription)
        print(' ReadAPI : ', Read_Number,' , TProb : ',TotalProb)

        img_dd = Image.open(n)
        strnum, classifiedStrNum =NDM(img_dd, count)

        DigitNumber2 = ''
        ExtractNumber =''
        RealProp = ''
        TrueFalse = ''
        #print('Read API ==>  ',count,)
        #Read_Number, TotalProb=readapi_3(img_dd, orp1.read_ocr_url, orp1.subscription)
        #print('Number from API ==>  ',Read_Number,'Prop from API ==> ',TotalProb)
     
    except:
        print(' ERROR somewhere')
    
    ##Summarize Analysis
    newrow= {'filename':filename,'Meter(Confidence)':TotalProb,'ReadAPI(ALL)':Read_Number, 'ReadAPI(Separated)':strnum, 'Classification':classifiedStrNum}
    df_Data = df_Data.append(newrow, ignore_index=True)
    print(' ==> ', df_Data)

## Display summary
print(' ==> ', df_Data)
print(' ======== Complete ============ ')
