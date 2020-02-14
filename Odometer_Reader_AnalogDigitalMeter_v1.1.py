import os
import glob
import numpy as np
import pandas as pd
import re
import OdoMeter_Reader_Parameters_v5 as orp1
from OdoMeter_Reader_Parameters_v5 import Digit_Detection as dd
from OdoMeter_Reader_Parameters_v5 import Number_Reader_ReadAPI_4_2
#from OdoMeter_Reader_Parameters_v5 import Number_Reader_Classification
from OdoMeter_Reader_Parameters_v5 import Order_Number
from OdoMeter_Reader_Parameters_v5 import Detect_Meter as dm
from OdoMeter_Reader_Parameters_v5 import Detect_Meter_2 as dm2
from OdoMeter_Reader_Parameters_v5 import Number_Reader_ReadAPI_3 as readapi_3
from OdoMeter_Reader_Parameters_v5 import Number_Reader_ReadAPI_3_1 as readapi_3_1
from OdoMeter_Reader_Parameters_v5 import Number_Reader_ReadAPI_3_4 as readapi_3_4
from OdoMeter_Reader_Parameters_v5 import Number_Reader_ReadAPI_3_5 as readapi_3_5
from OdoMeter_Reader_Parameters_v5 import IfLicensePlatTag
from OdoMeter_Reader_Parameters_v5 import MaskArea
from OdoMeter_Reader_Parameters_v5 import MaskArea_2

# Get file name and location
# Specify directory in which the images are kept
image_path="C:/Users/70018928/Documents/Project2020/TruckOdometer/TestDigital 86 precent/image_analogdigital/"

path=image_path+"*.jpg"
files = []
for file in glob.glob(path):
    files.append(file)
#print(' ==> ',files)

#Prepare table
df_Data=pd.DataFrame(columns = ['filename','Number', 'Meter(Confidence)','Extract Number','Confidence','Check', 'Meter', 'EX'])
#================================================================================
#Specify input path
count = 0
for n in files:

    # Extract Solution written as part of filename to use in output table for validation
    head, tail = os.path.split(n)
    filename = tail[:len(tail)-4]
    if (filename.find('t_')>1):
        start = filename.find('t_')
        RNumber = filename[start+2:]
    else:
        start = filename.find('[')
        end = filename.find(']')
        RNumber= filename[start+1:end]
    count = count+1
    print(' ==> ',filename ,' ==> ', RNumber)
    
    
    try:
        # Detect Digitbar on the meter (Analog or Digital meter) with Custom vision - Object detection
        print('Detect Meter ==> ',count,)
        img1, img_m, img_dd, area, prob, meterType=dm(n,orp1.dm_ocr_url,orp1.headers)
        # Remove any unwanted parts of picture which could mistakenly recognize as digitbar by applying gray patch on it
        # and re-detect the digit bar by Custom vision-object detection
        # This process repeats two times (there should be at most two unwanted parts to be removed)
        # The output is the cropped image of number bar
        if(IfLicensePlatTag(img_m)==1):
            img1, img_m, img_dd, area, image_masked=MaskArea(area, image_path)
        if(IfLicensePlatTag(img_m)==1):
            img1, img_m, img_dd, area=MaskArea_2(area, image_masked)
        DigitNumber2 = ''
        ExtractNumber =''
        RealProp = ''
        TrueFalse = ''
        print('Read API ==>  ',count,)
        # Cropped image sent to Custom vision - Batch Read API (OCR) to attempt to read any numbers in it
        # This 3-5 uses Black and White Binary Filter to process the image before calling the API
        Read_Number, TotalProb=readapi_3_5(img_dd, orp1.read_ocr_url, orp1.subscription, meterType)
        print('Number from API ==>  ',Read_Number,'Prop from API ==> ',TotalProb)
        ExtractNumber = Read_Number

        # If the image is of Digital Odometer, Use the output from 3_5
        # Otherwise for Analog Meter, checking if 3_5 can return full 6 digits answer 
        # or pass on to call 3_4 which uses Gray scale filter for preprocessing and call API
        if len(Read_Number) == 6 and meterType=="d_meter":
            ExtractNumber = Read_Number
            RealProp = TotalProb
            ifexcept='no'
            print('Decision ===>> API')
        else:
            Read_Number, TotalProb=readapi_3_4(img_dd, orp1.read_ocr_url, orp1.subscription, meterType)
            print('Read API without thresh_tozero ==>  ',Read_Number,'Prop from API ==> ',TotalProb)
            ExtractNumber = Read_Number
            ifexcept='yes'
    except:
            print(' ----------- ERROR SOMEWHERE ---------------')

            """Read_Number, TotalProb=readapi_3_1(n, orp1.read_ocr_url, orp1.subscription)
            print('Number from API ==>  ',Read_Number,'Prop from API ==> ',TotalProb)
            ExtractNumber = Read_Number
            if len(Read_Number) > 0:
                ExtractNumber = Read_Number
                RealProp = TotalProb
                print('Decision ===>> API')
            else:
                print('Run====>> API+Classification ==> ') 
                df_digit=dd(img_dd,orp1.mac_ocr_url,orp1.mac_headers)
                Digit, DProb=Number_Reader_ReadAPI_4_2(img_dd, orp1.read_ocr_url, orp1.subscription, df_digit,orp1.mac_ocr_url,orp1.mac_headers)
                DigitNumber, DigitProb =  Order_Number(df_digit, Digit, DProb)
                print('Number from API+Classification ==>  ',DigitNumber,'Prop from API ==> ',DigitProb)
                print('Decision ===>> API+Classification')
                if len(DigitNumber) > 5:
                    DigitNumber2 = DigitNumber[:6]
                ExtractNumber = DigitNumber2
                RealProp = DigitProb"""
    
    ##Summarize Analysis
    if ExtractNumber==RNumber :
        TrueFalse = 'True'
    else:
        TrueFalse = 'False'

    newrow= {'filename':filename,'Number':RNumber , 'Meter(Confidence)':prob,  'Extract Number':ExtractNumber , 'Confidence':RealProp, 'Check':TrueFalse, 'Meter':meterType,  'EX':ifexcept}
    df_Data = df_Data.append(newrow, ignore_index=True)
    print(' ==> ', df_Data)

## Display summary
print(' ==> ', df_Data)

## Specify output location to store the output table
file_path =r'C:\Users\70018928\Documents\Project2020\TruckOdometer\TestDigital 86 precent\Odometer_DigitalMeter_Test_2\Check_Output\COutput_Testmile2.csv'
df_Data.to_csv(file_path)
