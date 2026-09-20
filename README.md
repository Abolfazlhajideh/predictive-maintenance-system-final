# Predictive Maintenance System

A machine-learning-based system for monitoring the condition of industrial equipment.

The project analyzes simulated sensor data and detects abnormal equipment behavior using:

- Temperature
- Temperature change
- Pressure
- Pressure change
- Vibration
- Vibration change
- Date and time information

## Project Goal

The goal of this project is to detect abnormal operating conditions before they develop into serious equipment faults.

The system classifies equipment behavior into three states:

- Normal
- Warning
- Danger

## System Workflow

```text
Sensor data
    ↓
Data processing
    ↓
Feature calculation
    ↓
Machine-learning model
    ↓
Status prediction
    ↓
Warning and result storage
```

## Machine-Learning Model

The project uses the Isolation Forest algorithm for anomaly detection.

The model is trained using the following features:

```text
temperature
temperature_change
pressure
pressure_change
vibration
vibration_change
```

The trained model is saved as:

```text
models/isolation_forest_model.pkl
```

## Project Structure

```text
predictive-maintenance-system/
│
├── data/
│   ├── raw/
│   │   └── sensor_data.csv
│   └── processed/
│
├── models/
│   └── isolation_forest_model.pkl
│
├── notebooks/
│
├── results/
│   ├── figures/
│   │   ├── confusion_matrix.png
│   │   ├── device_status.png
│   │   └── live_monitoring.png
│   ├── live_predictions.csv
│   └── predictions.csv
│
├── src/
│   ├── data_processing.py
│   ├── evaluate_model.py
│   ├── feature_engineering.py
│   ├── plot_live_results.py
│   ├── realtime_prediction.py
│   └── train_model.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

## Installation

Create and activate a virtual environment:

```bash
python -m venv venv
```

On Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

Install the required libraries:

```bash
python -m pip install -r requirements.txt
```

## Run the Project

Generate sensor data:

```bash
python src/data_processing.py
```

Train the anomaly-detection model:

```bash
python src/train_model.py
```

Evaluate the model:

```bash
python src/evaluate_model.py
```

Run simulated live monitoring:

```bash
python src/realtime_prediction.py
```

Create the monitoring figure:

```bash
python src/plot_live_results.py
```

## Model Evaluation

The current model was tested on simulated data.

The evaluation produced:

- Accuracy: 0.88
- Normal recall: 0.93
- Danger recall: 0.71

These results are based on simulated data and should not be considered real industrial performance.

## Current Limitations

- The sensor data is simulated.
- The project has not yet been connected to a physical sensor board.
- The model has not been tested on a real industrial machine.
- The warning thresholds require calibration using real equipment data.

## Future Development

- Connect Arduino or ESP32 sensors.
- Collect real temperature, pressure and vibration data.
- Improve feature engineering.
- Reduce false alarms.
- Add a real-time dashboard.
- Add email or notification alerts.
- Test the system on real industrial equipment.

## Author

Mechanical engineering student project focused on predictive maintenance, smart automation and machine learning
