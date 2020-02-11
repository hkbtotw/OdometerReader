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

# Get file name and location
# Specify directory in which the images are kept
image_path="C:/Users/70018928/Documents/GitHub/OdometerReader/TestImage/"

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
        print('Detect Meter ==> ',count,)
        img1, img_m, img_dd, area, prob=dm(n,orp1.dm_ocr_url,orp1.headers)
        if(IfLicensePlatTag(img_m)==1):
            img1, img_m, img_dd, area, image_masked=MaskArea(area, image_path)
        if(IfLicensePlatTag(img_m)==1):
            img1, img_m, img_dd, area=MaskArea_2(area, image_masked)

        filename=image_path_output+str(count)+'.jpg'
        img_dd.save(filename, quality=100)


        Read_Number, TotalProb=readapi_3(img_dd, orp1.read_ocr_url, orp1.subscription, count, count)
        print(' ReadAPI : ', Read_Number,' , TProb : ',TotalProb)

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
file_path =r'C:\Users\70018928\Documents\GitHub\OdometerReader\Output_TestImage\Output.csv'
df_Data.to_csv(file_path)

print(' ======== Complete ============ ')
