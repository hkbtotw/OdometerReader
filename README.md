Complete Odometer Reader Program

Computer Vision Code: Read Numbers from Truck Odometer

Odometer_Reader_AnalogDigitalMeter_v1.0.py reads Images in TestImage and detect and predict numbers in the Odometer images. Odometer_Reader_Parameter_v6 contains Parameters and Functions used in the main code

(Please note that the subscription was taken off and this information could be found in the local drive)

Step:
1.Code reads the image from local drive

2.Number bar detection by Object Recognition (Cognitive Service) 

3.Image processing (Filtering) used to process the image (Grayscale and Binary used for Analog Meter and Binary used for Digital Meter)

4.Batch Read API (Cognitive Service) reads the number out of the meter image

Sample Images:
Analog Meter
<img src=https://github.com/hkbtotw/OdometerReader/blob/master/TestImage/IMG_9090_[052334]_A1.JPG alt="Analog" width="200"/>

Digital Meter:
<img src=https://github.com/hkbtotw/OdometerReader/blob/master/TestImage/IMG_9028[278168].JPG alt="Digital" width="200"/>
