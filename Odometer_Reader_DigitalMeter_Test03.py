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
from OdoMeter_Reader_Parameters_v4 import Detect_Digital_Number_Bar as DDNB

# Get file name and location

# Specify directory in which the images are kept
image_path="C:/Users/70018928/Documents/GitHub/OdometerReader/DigitalTestImage/"

image_path_output="C:/Users/70018928/Documents/GitHub/OdometerReader/Output_TestImage/"

path=image_path+"*.jpg"
files = []
for file in glob.glob(path):
    files.append(file)
#print(' ==> ',files)

#Prepare table
df_Data=pd.DataFrame(columns = ['filename','Number', 'Meter(Confidence)','Extract Number','Confidence','Check'])
#================================================================================
#Specify input path
count = 0
for n in files:
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
    
    # Detect Digitbar on the meter (Analog or Digital meter) with Custom vision - Object detection

    try:
        print('Detect Meter ==> ',count,)
        img1, img_m, img_dd, area, prob=dm(n,orp1.dm_ocr_url,orp1.headers)
        if(IfLicensePlatTag(img_m)==1):
            img1, img_m, img_dd, area, image_masked=MaskArea(area, image_path)
        if(IfLicensePlatTag(img_m)==1):
            img1, img_m, img_dd, area=MaskArea_2(area, image_masked)
        DigitNumber2 = ''
        ExtractNumber =''
        RealProp = ''
        TrueFalse = ''
        print('Read API ==>  ',count,)

        #img1, img_m, img_dd, area, prob=DDNB(img_dd,orp1.dgm_ocr_url,orp1.headers)
        
        # Test run 
        #Read_Number, TotalProb=readapi_3_1(n, orp1.read_ocr_url, orp1.subscription)
        
        # Normal Run 
        #Read_Number, TotalProb=readapi_3(img_dd, orp1.read_ocr_url, orp1.subscription, count, count)
        Read_Number, TotalProb =readapi_3(img_dd, orp1.read_ocr_url, orp1.subscription, count, count)

        print('Number from API ==>  ',Read_Number,'Prop from API ==> ',TotalProb)
        ExtractNumber = Read_Number
        #prob=TotalProb
        #RealProp=TotalProb

        """
        if len(Read_Number) > 0:
            ExtractNumber = Read_Number
            RealProp = TotalProb
            print('Decision ===>> API+Classification')
        else:
            print('Run====>> API+Classification ==> ') 
            df_digit=dd(img_dd,orp1.mac_ocr_url,orp1.mac_headers)
            Digit, DProb=Number_Reader_ReadAPI_4_2(img_dd, orp1.read_ocr_url, orp1.subscription, df_digit,orp1.mac_ocr_url,orp1.mac_headers)
            DigitNumber, DigitProb =  Order_Number(df_digit, Digit, DProb)
            print('Number from API+Classification ==>  ',DigitNumber,'Prop from API ==> ',DigitProb)
            print('Decision ===>> API')
            if len(DigitNumber) > 5:
                DigitNumber2 = DigitNumber[:6]
            ExtractNumber = DigitNumber2
            RealProp = DigitProb
        """   
    except:
        """
        Read_Number, TotalProb=readapi_3_1(n, orp1.read_ocr_url, orp1.subscription)
        print('Number from API ==>  ',Read_Number,'Prop from API ==> ',TotalProb)
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
            RealProp = DigitProb
        """
        print(' ERROR somewhere')
    
    ##Summarize Analysis
    if ExtractNumber==RNumber :
        TrueFalse = 'True'
    else:
        TrueFalse = 'False'

    newrow= {'filename':filename,'Number':RNumber , 'Meter(Confidence)':prob,  'Extract Number':ExtractNumber , 'Confidence':RealProp, 'Check':TrueFalse}
    df_Data = df_Data.append(newrow, ignore_index=True)
    print(' ==> ', df_Data)

## Display summary
print(' ==> ', df_Data)

file_path =r'C:\Users\70018928\Documents\GitHub\OdometerReader\Output_TestImage\Output.csv'
df_Data.to_csv(file_path)

print(' ========= Complete ========== ')
