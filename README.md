# Battery Current Prediction Model Training
This directory contains the training pipeline used for battery current prediction and state classification on Android devices.

## Overview
The objective of the project is to:
1. Generate Android device activity state labels using clustering technique.
2. Classify Android device activity state.
3. Predict battery current consumption one minute ahead using time-series forecasting.
4. Deploy the trained models to Android devices using TensorFlow Lite.

## Requirements
```bash
  pip install -r requirements.txt
```

## Training

### Preprocessing
```bash
  python -m Clustering.src.preprocessing.preprocessing
```

### Train Clustering Model
```bash
  python -m Clustering.src.model.train  
```

### Train Classification Model
```bash
  # Train Test Split
  python -m Classification.src.preprocessing.tts

  # Softmax Regression
  python -m Classification.src.model.train_lr
  
  # ANN
  python -m Classification.src.model.train    
```

### Train Time Series Model
```bash
  # RNN
  python -m TimeSeries.src.model.train_rnn
  
  # LSTM
  python -m TimeSeries.src.model.train_lstm
  
  # GRU
  python -m TimeSeries.src.model.train_gru
  
  # CNN-RNN
  python -m TimeSeries.src.model.train_cnn_rnn
  
  # CNN-LSTM
  python -m TimeSeries.src.model.train_cnn_lstm
  
  # CNN-GRU
  python -m TimeSeries.src.model.train_cnn_gru
```

### Convert to TFLite
```bash
  # Classification
  python -m Classification.src.convert.to_tflite
  
  # Time Series
  python -m TimeSeries.src.convert.to_tflite
```

## Dataset
The dataset was collected from Android smartphone with a sampling interval of 4 seconds.

### Features
- CPU Usage
- Memory Usage
- Temperature
- Battery Level
- Battery Current
- Battery Charging State
- Upload Speed
- Download Speed
- Screen State
- Brightness

### Time Series Configuration

| Parameter           | Value      |
|---------------------|------------|
| Sampling Interval   | 4 seconds  |
| Window Size         | 45 samples |
| Horizon Size        | 15 Samples |
| Historical Duration | 3 minutes  |
| Predicted Horizon   | 1 minute   |

## Model Configuration

### Clustering

Prior to classification, activity state labels were generated using unsupervised clustering method.

#### Clustering Pipeline 
1. Feature Scaling using StandardScaler.
2. Dimensionality Reduction using PCA.
3. Clustering using K-Means or DBSCAN.
4. Cluster interpretation and label assignment.

#### K-Means

The K-Means clustering algorithm was configured using the following configuration:

| Configuration     | Value                                                                         |
|-------------------|-------------------------------------------------------------------------------|
| Number of Cluster | 3                                                                             |
| Random State      | 42                                                                            |
| Feature Type      | Numerical Feature                                                             |
| Evaluation Metric | - Silhouette Score <br/>- Davies-Bouldin Index <br/>- Calinski-Harabasz Index |

#### DBSCAN

The DBSCAN clustering algorithm was configured using the following configuration:

| Parameter         | Value                                                                         |
|-------------------|-------------------------------------------------------------------------------|
| Epsilon (eps)     | 0.5                                                                           |
| Min Samples       | 5                                                                             |
| Feature Type      | Numerical Features                                                            |
| Evaluation Metric | - Silhouette Score <br/>- Davies-Bouldin Index <br/>- Calinski-Harabasz Index |

### Classification

The classification model is used to predict the state of the Android devices using the labels gathered from clustering process.

#### Softmax Regression

The Softmax Regression was configured with the following architecture:

| Layer        | Configuration      |
|--------------|--------------------|
| Input Layer  | 7 Features         |
| Output Layer | 3 Neurons, Softmax |

Training Parameters:

| Parameter      | Value                           |
|----------------|---------------------------------|
| Optimizer      | Adam                            |
| Loss Function  | Sparse Categorical Crossentropy |
| Batch Size     | 32                              |
| Epochs         | 100                             |
| Learning Rate  | 0.001                           |
| Early Stopping | Enabled                         |

#### ANN

The ANN was configured with the following architecture

| Layer        | Configuration      |
|--------------|--------------------|
| Input Layer  | 7 Features         |
| Dense 1      | 32 Neurons         |
| Output Layer | 3 Neurons, Softmax |

Training Parameters:

| Parameter      | Value                           |
|----------------|---------------------------------|
| Optimizer      | Adam                            |
| Loss Function  | Sparse Categorical Crossentropy |
| Batch Size     | 32                              |
| Epochs         | 100                             |
| Learning Rate  | 0.001                           |
| Early Stopping | Enabled                         |

### Time Series

The forecasting models were used to predict battery current consumption for the next one minute using three minutes of historical observations.

Input Shape:
(45, 9)

Output Shape:
(15,)

The output represents the battery current predictions for the next 15 timestamps (1 minute ahead).
#### RNN

The RNN forecasting model was configured as follows: 

| Layer | Configuration |
|-------|---------------|
| Input | (45, 9)       |
| RNN 1 | 10 Units      |
| RNN 2 | 10 Units      |
| Dense | 15 Outputs    |

Training Parameters:

| Parameter      | Value   |
|----------------|---------|
| Optimizer      | Adam    |
| Loss Function  | MSE     |
| Batch Size     | 32      |
| Epochs         | 100     |
| Learning Rate  | 0.001   |
| Early Stopping | Enabled |

#### LSTM

The LSTM forecasting model was configured as follows:

| Layer  | Configuration |
|--------|---------------|
| Input  | (45, 9)       |
| LSTM 1 | 10 Units      |
| LSTM 2 | 5 Units       |
| Dense  | 15 Outputs    |

Training Parameters:

| Parameter      | Value   |
|----------------|---------|
| Optimizer      | Adam    |
| Loss Function  | MSE     |
| Batch Size     | 32      |
| Epochs         | 100     |
| Learning Rate  | 0.001   |
| Early Stopping | Enabled |

#### GRU

The GRU forecasting model was configured as follows:

| Layer | Configuration |
|-------|---------------|
| Input | (45, 9)       |
| GRU 1 | 10 Units      |
| GRU 2 | 10 Units      |
| Dense | 15 Outputs    |

Training Parameters:

| Parameter      | Value   |
|----------------|---------|
| Optimizer      | Adam    |
| Loss Function  | MSE     |
| Batch Size     | 32      |
| Epochs         | 100     |
| Learning Rate  | 0.001   |
| Early Stopping | Enabled |

#### CNN-RNN

The CNN-RNN forecasting model was configured as follows:

| Layer        | Configuration              |
|--------------|----------------------------|
| Input        | (45, 9)                    |
| Conv1D 1     | 4 Filters, Kernel Size = 3 |
| Conv1D 2     | 8 Filters, Kernel Size = 3 |
| Conv1D 3     | 4 Filters, Kernel Size = 3 |
| MaxPooling1D | Pool Size = 2              |
| RNN 1        | 5 Units                    |
| RNN 2        | 5 Units                    |
| Dense        | 15 Outputs                 |

Training Parameters:

| Parameter      | Value   |
|----------------|---------|
| Optimizer      | Adam    |
| Loss Function  | MSE     |
| Batch Size     | 32      |
| Epochs         | 100     |
| Learning Rate  | 0.001   |
| Early Stopping | Enabled |

#### CNN-LSTM

The CNN-LSTM forecasting model was configured as follows:

| Layer        | Configuration              |
|--------------|----------------------------|
| Input        | (45, 9)                    |
| Conv1D 1     | 4 Filters, Kernel Size = 3 |
| Conv1D 2     | 8 Filters, Kernel Size = 3 |
| Conv1D 3     | 4 Filters, Kernel Size = 3 |
| MaxPooling1D | Pool Size = 2              |
| LSTM 1       | 5 Units                    |
| LSTM 2       | 5 Units                    |
| Dense        | 15 Outputs                 |

Training Parameters:

| Parameter      | Value   |
|----------------|---------|
| Optimizer      | Adam    |
| Loss Function  | MSE     |
| Batch Size     | 32      |
| Epochs         | 100     |
| Learning Rate  | 0.001   |
| Early Stopping | Enabled |

#### CNN-GRU

The CNN-GRU forecasting model was configured as follows:

| Layer        | Configuration              |
|--------------|----------------------------|
| Input        | (45, 9)                    |
| Conv1D 1     | 4 Filters, Kernel Size = 3 |
| Conv1D 2     | 8 Filters, Kernel Size = 3 |
| Conv1D 3     | 4 Filters, Kernel Size = 3 |
| MaxPooling1D | Pool Size = 2              |
| GRU 1        | 5 Units                    |
| GRU 2        | 5 Units                    |
| Dense        | 15 Outputs                 |

Training Parameters:

| Parameter      | Value   |
|----------------|---------|
| Optimizer      | Adam    |
| Loss Function  | MSE     |
| Batch Size     | 32      |
| Epochs         | 100     |
| Learning Rate  | 0.001   |
| Early Stopping | Enabled |

## Experimental Results

### Clustering

| Model         | Silhouette | Davies-Bouldin | Calinski-Harabasz |
|---------------|------------|----------------|-------------------|
| K-Means       | 0.248      | 1.482          | 10392.81          |
| K-Means + PCA | 0.453      | **0.784**      | **32023.03**      |
| DBSCAN        | -0.419     | 1.414          | 88.88             |
| DBSCAN + PCA  | **0.592**  | 1.176          | 32.59             |

Based on the table above, it can be concluded that K-Means + PCA has the best overall performance out of all the models trained.

Cluster characteristics:

| Cluster | CPU    | Memory | Temperature | Battery Current | Upload    | Download   |
|---------|--------|--------|-------------|-----------------|-----------|------------|
| 0       | 38.25% | 72.70% | 33.33°C     | -192.42 mA      | 1.07 kbps | 3.28 kbps  |
| 1       | 56.18% | 77.56% | 35.02°C     | -356.06 mA      | 1.40 kbps | 7.13 kbps  |
| 2       | 49.54% | 74.87% | 33.91°C     | -301.60 mA      | 7.08 kbps | 60.53 kbps |

Cluster labels were manually interpreted and mapped to the following activity states:
- Class 0 (Low Activity)
- Class 1 (High Computational Activity)
- Class 2 (Network Activity)

### Classification

| Model              | Accuracy | Precision | Recall   | F1 Score |
|--------------------|----------|-----------|----------|----------|
| Softmax Regression | 0.86     | 0.86      | 0.86     | 0.86     |
| ANN                | **0.97** | **0.98**  | **0.97** | **0.97** |

### Time Series

| Model    | MAE       | MAPE       | MSE       | RMSE      | R²        |
|----------|-----------|------------|-----------|-----------|-----------|
| RNN      | 0.627     | 111.716    | 0.870     | 0.933     | 0.309     |
| LSTM     | **0.282** | 31.986     | 0.477     | 0.690     | 0.621     |
| GRU      | 0.437     | 105.071    | 0.483     | 0.649     | 0.617     |
| CNN-RNN  | 0.744     | 260.027    | 1.416     | 1.190     | -0.124    |
| CNN-LSTM | 0.744     | 77.441     | 0.515     | 0.781     | 0.591     |
| CNN-GRU  | 0.744     | **17.124** | **0.319** | **0.565** | **0.746** |

### Deployment

#### Classification
| Model   | Location | Class     | CPU    | Mem    | Temp    | Current    | Up         | Down        |
|---------|----------|-----------|--------|--------|---------|------------|------------|-------------|
| Softmax | Device   | Idle      | 40.19% | 70.68% | 35.01°C | 256.14 mA  | 1.23 kbps  | 4.01 kbps   |
| Softmax | Device   | High Comp | 59.72% | 71.40% | 34.75°C | -293.03 mA | 1.57 kbps  | 3.91 kbps   |
| Softmax | Device   | Network   | 49.01% | 72.05% | 35.41°C | -102.31 mA | 17.27 kbps | 494.95 kbps |
| ANN     | Device   | Idle      | 37.68% | 69.77% | 32.97°C | -161.22 mA | 1.39 kbps  | 2.07 kbps   |
| ANN     | Device   | High Comp | 62.58% | 73.19% | 35.18°C | -419.30 mA | 5.53 kbps  | 77.48 kbps  |
| ANN     | Device   | Network   | 50.93% | 72.21% | 35.86°C | 154.61 mA  | 16.26 kbps | 85.10 kbps  |
| Softmax | Server   | Idle      | 37.25% | 71.03% | 37.51°C | 423.67 mA  | 2.58 kbps  | 4.54 kbps   |
| Softmax | Server   | High Comp | 58.30% | 72.41% | 37.23°C | -396.43 mA | 3.02 kbps  | 9.50 kbps   |
| Softmax | Server   | Network   | 49.11% | 72.17% | 37.47°C | -264.45 mA | 12.17 kbps | 384.04 kbps |
| ANN     | Server   | Idle      | 37.89% | 71.2%  | 34.81°C | -227.86 mA | 2.53 kbps  | 3.03 kbps   |
| ANN     | Server   | High Comp | 51.63% | 73.31% | 35.57°C | -347.92 mA | 11.02 kbps | 342.51 kbps |
| ANN     | Server   | Network   | 45.86% | 72.44% | 36.41°C | 256.5 mA   | 6.79 kbps  | 108.86 kbps |

#### Time Series

| Model   | Location | Note         | MSE          | RMSE      | R²      |
|---------|----------|--------------|--------------|-----------|---------|
| GRU     | Device   | All          | 217121.4493  | 465.9629  | 0.6556  |
| GRU     | Device   | Charging     | 770733.0948  | 877.9141  | 0.3036  |
| GRU     | Device   | Not Charging | 118534.1117  | 344.2878  | -0.0623 |
| CNN-GRU | Device   | All          | 357056.2467  | 597.5418  | 0.6244  |
| CNN-GRU | Device   | Charging     | 1916382.2821 | 1384.3346 | -0.1111 |
| CNN-GRU | Device   | Not Charging | 129371.8977  | 359.6831  | 0.0231  |
| GRU     | Server   | All          | 272398.5917  | 521.9182  | 0.7119  |
| GRU     | Server   | Charging     | 915107.1348  | 956.6123  | 0.2979  |
| GRU     | Server   | Not Charging | 140757.9368  | 375.1772  | 0.0017  |
| CNN-GRU | Server   | All          | 102782.1380  | 320.5965  | 0.7090  |
| CNN-GRU | Server   | Charging     | 358260.8108  | 598.5489  | -0.5351 |
| CNN-GRU | Server   | Not Charging | 64935.2133   | 254.8239  | -0.0715 |

#### Inference Time

| Model   | Location | Mean      | Min      | Max          | Q1        | Q3        |
|---------|----------|-----------|----------|--------------|-----------|-----------|
| Softmax | Device   | 8.166ms   | 0.693ms  | 8981.641ms   | 2.253ms   | 12.249ms  |
| ANN     | Device   | 21.003ms  | 0.665ms  | 145260.467ms | 2.156ms   | 13.361ms  |
| Softmax | Server   | 203.407ms | 88.248ms | 11903.947ms  | 126.28ms  | 223.229ms |
| ANN     | Server   | 206.06ms  | 74.411ms | 10092.492ms  | 130.509ms | 220.026ms |

| Model   | Location | Mean      | Min      | Max         | Q1        | Q3        |
|---------|----------|-----------|----------|-------------|-----------|-----------|
| GRU     | Device   | 2.701ms   | 0.526ms  | 22.897ms    | 1.359ms   | 4.112ms   |
| CNN-GRU | Device   | 1.883ms   | 0.345ms  | 49.66ms     | 0.881ms   | 2.818ms   |
| GRU     | Server   | 246.688ms | 96.027ms | 10902.657ms | 165.792ms | 275.522ms |
| CNN-GRU | Server   | 246.058ms | 96.33ms  | 7759.604ms  | 167.058ms | 273.358ms |

#### Application Overhead

| Scenario       | CPU Usage | Memory Usage | Temperature |
|----------------|-----------|--------------|-------------|
| Normal         | 40.16%    | 67.84%       | 32.80°C     |
| Classification | 40.78%    | 63.31%       | 35.08°C     |
| Time Series    | 43.12%    | 66.24%       | 37.26       |

## Result Summary

| Experiment              | Best Configuration   | Key Result                                |
|-------------------------|----------------------|-------------------------------------------|
| Clustering              | PCA + K-Means        | Best clustering quality                   |
| Classification          | ANN                  | Highest classification performance        |
| Time Series Forecasting | CNN-GRU              | RMSE = 0.565, R² = 0.746                  |
| Deployment              | TensorFlow Lite      | Lower inference latency                   |
| Application Overhead    | On-Device Inference  | Low CPU, memory, and temperature overhead |