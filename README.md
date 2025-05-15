# TLCellClassifier

# Introduction
This repository is the implementation for paper "TLCellClassifier: Machine Learning-Based Cell Classification for Bright-Field Time-Lapse Images".

This application contains three components: Cell Detection, Cell Tracking and Cell Classification.
The overall workflow of the framework is as follows:
![workflow](https://github.com/QibingJiang/cell_classification_ml/assets/63761274/cfb8c053-0bf1-4afb-942b-cdca2e66070e)

The result of the proposed method compared with KTH-SE\(Cell tracking Chellenge\) and YOLO V8 is as followed:

![Beacon_A](https://github.com/QibingJiang/cell_classification_ml/assets/63761274/3282ad40-496c-46cb-a4a8-7750ac17ef3e)

# Running
The work depends on 
```
pip install opencv_python
pip install scipy
pip install astropy
pip install shutil
pip install tensorflow
```
Run the detection, tracking and classification by the command:
```
python3 cell_process.py /input_dir/ /output_dir/
```
The input_dir is the folder where the original images are. The output_dir is the folder where TLCellClassifier put processing result.

# Training
## Cell Detection
To train the cell Detection model, the dataset as follows should be prepared.


<img src="https://github.com/compbiolabucf/TLCellClassifier/blob/main/assets/detection.png" width="300">

There are two classes in the CNN-based classifier. One is cells, and the other is non-cells. 10x10 images are cropped and labelled.

After the dataset is prepared, just run the following code:

```
python3 dig_classify_v1.py
```

## Cell Tracking
To train cell tracking model, the images of cell tracks should be prepared. 18x18 images of cell neighbourhood are cropped across time points and labelled.

<img src="https://github.com/compbiolabucf/TLCellClassifier/blob/main/assets/Tracking.png" width="500">

You can train the model by the following command:
```
python3 train_cells.py \
    --dataset_dir=./cells \
    --loss_mode=cosine-softmax \
    --log_dir=./output/cells/ \
    --run_id=cosine-softmax
```

## Cell Classification
Cell Classification is implemented by applying LSTM on videos of individual cells.
<video src="https://github.com/compbiolabucf/TLCellClassifier/blob/main/assets/Media1.avi" width="100" controls></video>

You can train the model by running the following command:
```
python3 video_classification_splits_error_augment.py
```
