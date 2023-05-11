# Har-handicaps
<img src="https://img.shields.io/badge/Python-blue?logo=Python&logoColor=white"/> <img src="https://img.shields.io/badge/Zenodo-FF9E0F"/> <img src="https://img.shields.io/github/languages/code-size/beygee/survive"/> <img src="https://img.shields.io/badge/Smart Phone-green"/> <img src="https://img.shields.io/badge/Smart Watch-yellow"/>  
**Official database and implementation of Ubicomp 2023 paper  
(Human Activity Recognition of Pedestrians with Mobility Disabilities)**

Here are download link for our dataset and  preprocessing and classification analysis codes.
We present the analysis results through several basic ML & DL models.

![1](https://github.com/myLABtemp/HAR-handicaps/assets/81300282/55a0f356-6bdd-4722-a58e-625582c247e1)


## Dependencies
- Python 3.7
- Numpy X.X
- Pandas X.X
- Tensorflow 2.X.X

## Dataset Setup

Please download the dataset from [here](https://127.0.0.1)  
To analyize the dataset by our demo, set the dataset file like below
```bash
${Repositiory_ROOT}/
|-- 1_data_raw
|   |-- 5001
|   |-- 5001-2
|   |-- 5002
|   |-- ...
|   |-- 5127-2
```

## Preprocessing Data
This process consists of two stages which are resampling and slincing data with appropriate window size.

First, we resampling the smart phone & smart watch sensor data uniformly. Although we used same frequency rates to save data.
Although we used the same frequency rate for all conditions (devices, classes, participants...etc), not all data were recorded uniformly due to technical limitations.
In addition, since there is a slight delay in the start time of data recording between the smartphone and the watch, it is necessary to adjust it.

To resampling data:

```bash
python preprocessing/valid_data_interpolation.py
```

Then the folder "valid_phone_watch_data" are created and all resampling data are generated below it. 
```bash
${Repositiory_ROOT}/
|-- valid_phone_watch_data
|   |-- 5001
|   |-- 5001-2
|   |-- ...
```

Second, transform the dataset for ML & DL analysis. In this step, we sliced data per 5s without overlapping  

```bash
python preprocessing/valid_data_tonpy.py
```

You can confirm the processed data and auto generated labels.  
The shape of data : ( 600/5(window size) sec x 60 hz x N participants, 5(window size), 3 sensor x 3 axis )  
The shape of label : ( 600/5(window size) sec x 60 hz x N participants) {'still': 0, 'walking': 1, 'crutches': 2, 'walker': 3, 'manual': 4, 'motorized': 5}  

```bash
valid_phone_watch_data/5s_valid_data_obj.npy
valid_phone_watch_data/5s_valid_label_obj.npy
```

## ML & DL Analysis & Results
You can simply get the result by runnung the code 

```bash
python main.py
```

As a result, confusion matrix, model, classification reports with precision, recall, and f1score for each class are saved in the following directory.
Results are provided for each scenario, sensor data, and model type.

* Number of Sensor : `A` , `AG` , `AGM`  { A: linear accelerometer, G: gyroscope, M: magnetometer }  
* Type of Model :`DT`, `RF`, `SVM`, `Multi_CNN`, `multi_LSTM`, `XGB`, `Transformer`  
* Type of scenario :  

<table style="text-align:center" width="900" >
  <tr style>
    <th width="300">Classification goal</th>
    <th width="100">Still</th>
    <th width="100">Walking</th>
    <th width="100">crutches</th>
    <th width="100">Walker</th>
    <th width="100">Manual wheelchair</th>
    <th width="100">Electric wheelchair</th>
  </tr>
   <tr style="text-align:center">
    <td><code>Mobility disability</code></td>
    <td>Still</td>
    <td>Walking</td>
    <td colspan="4">Mobility disability</td>
  </tr>
  <tr style="text-align:center">
    <td><code>Walking aids & wheelchairs</code></td>
    <td>Still</td>
    <td>Walking</td>
    <td colspan="2">Walking assistive devices</td>
    <td colspan="2">Wheelchairs</td>
  </tr>
   <tr style="text-align:center">
    <td><code>Mobility aids in detail</code></td>
    <td>Still</td>
    <td>Walking</td>
    <td>Crutches</td>
    <td>Walker</td>
    <td>Manual wheelchair</td>
    <td>Electric wheelchair</td>
  </tr>
</table>
  
<br/>

***  
<br/>  
The Result file  
<br/>
<br/>

```bash
HAR_result/{num_of_sensors}_sensors_X_{scenario}_{slicing_time}sec_{model_type}_{split}_result-{fold}.txt" // classification_reports
HAR_result/{num_of_sensors}_sensors_X_{scenario}_{slicing_time}sec_{model_type}_{split}.t" // model
HAR_result/{num_of_sensors}_sensors_X_{scenario}_{slicing_time}sec_{model_type}_{split}_model-{fold}.png" // confusion matrix
```

<br/>

The Result Table  
![2](https://github.com/myLABtemp/HAR-handicaps/assets/81300282/8a7463c9-a035-438d-8086-9efddfbf8582)
  
  
 
More detail results including all conditions ( scenario, sensor data, model type, split type ) are described in [our paper](https://127.0.0.1)  


## Citation

```BibTeX
@InProceedings{not yet published,
    author    = {YEJI WOO, SUNGJIN HWANG, SUNGYOON LEE, YOUNGWUG CHO, HANSUNG KIM, JAEHYUK CHA, KWANGUK (KENNY) KIM*},
    title     = {Human Activity Recognition of Pedestrians with Mobility Disabilities},
    booktitle = {Proceedings of the 2023 ACM Conference on Pervasive and Ubiquitous Computing (Ubicomp)},
    month     = {not yet published},
    year      = {2023},
    pages     = {not yet published}
}
```

