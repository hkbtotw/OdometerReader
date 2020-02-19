<h1>Odometer Reader Program</h1>
(Complete Prototype Phase on 20200218)
<h2>Computer Vision Code: Read Numbers from Truck Odometer</h2>

<i>Odometer_Reader_AnalogDigitalMeter_v1.0.py reads Images in TestImage and detect and predict numbers in the Odometer images. Odometer_Reader_Parameter_v6 contains Parameters and Functions used in the main code</i>

(Please note that the subscription was taken off and this information could be found in the local drive)

<b>Step:</b><br />
1.Code reads the image from local drive<br />
2.Number bar detection by Object Recognition (Cognitive Service) <br />
3.Image processing (Filtering) used to process the image (Grayscale and Binary used for Analog Meter and Binary used for Digital Meter) <br />
4.Batch Read API (Cognitive Service) reads the number out of the meter image <br />

<h2>Prediction Performance</h2>
<Table Border=4>
<Caption>  Prototype (Tested on 20200219 - Iteration 3 : Model " Detect Meter" on  customvision.ai)  </Caption>
<Tr>
      <th>Prediction Model</th>
      <th>Analog - Precision</th>
      <th>Analog - Recall</th>
      <th>Digital - Precision</th>
      <th>Digital - Recall</th>
</Tr>
<Tr>
      <th>Object Detection</th>
      <th>100%</th>
      <th>100%</th>
      <th>96.6%</th>
      <th>90.3%</th>
</Tr>
<Tr>
      <th>OCR (Read API) - used as is from MicroSoft</th>
      <th>-</th>
      <th>-</th>
      <th>-</th>
      <th>-</th>
</Tr>
</Table>


<b>Sample Images:<br/></b>
Analog Meter

<img src=https://github.com/hkbtotw/OdometerReader/blob/master/TestImage/IMG_9090_[052334]_A1.JPG alt="Analog" width="200"/>

Detected Analog Digitbar by Object Detection

<img src=https://github.com/hkbtotw/OdometerReader/blob/master/TestImage/DetectedAnalog.jpg alt="Detected Analog" width="400"/>

Digital Meter:

<img src=https://github.com/hkbtotw/OdometerReader/blob/master/TestImage/IMG_9028[278168].JPG alt="Digital" width="200"/>

Detected Digital Digitbar by Object Detection

<img src=https://github.com/hkbtotw/OdometerReader/blob/master/TestImage/DetectedDigital.jpg alt="Detected Digital" width="400"/>
