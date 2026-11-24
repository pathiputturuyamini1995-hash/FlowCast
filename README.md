FlowCast – Smart Traffic Flow Prediction System

1. Project Overview

FlowCast is an AI-powered Smart Traffic Flow Prediction System developed using Machine Learning and Deep Learning techniques.

The system analyses historical traffic sensor data, weather observations, and calendar events to understand traffic patterns, predict traffic volume, classify congestion levels, and estimate accident risk.

An interactive Streamlit dashboard is developed to visualise traffic information and display prediction results.

2. Project Objectives

- Analyse historical traffic patterns.
- Predict future traffic volume using Machine Learning and Deep Learning.
- Identify different traffic congestion levels.
- Estimate accident risk for the next 60 minutes.
- Analyse the relationship between weather conditions and traffic.
- Provide interactive traffic visualisations through a Streamlit dashboard.

3. Technologies Used

- Programming Language: Python
- Data Analysis: Pandas, NumPy
- Machine Learning: Scikit-learn
- Deep Learning: TensorFlow, Keras
- Data Visualisation: Matplotlib
- Development Environment: Jupyter Notebook, VS Code
- Dashboard: Streamlit

4. Datasets Used

The project uses the following datasets:

1. "traffic_sensor_log.csv" – Historical traffic sensor information.
2. "weather_observations.csv" – Weather-related observations.
3. "calendar_events.csv" – Calendar and event information.

These datasets are used for traffic analysis, preprocessing, feature engineering, and model development.

5. Machine Learning and Deep Learning Models

The following models were implemented:

5.1 Random Forest Regressor

Used to predict traffic volume based on historical traffic patterns and engineered features.

5.2 LSTM (Long Short-Term Memory)

Used for time-series traffic volume forecasting based on previous traffic observations.

5.3 Congestion Classification Model

Used to classify traffic congestion levels based on traffic-related features.

5.4 Extra Trees Classifier

Used to estimate the possibility of an accident occurring within the next 60 minutes.

6. Model Performance

Traffic Volume Prediction – Random Forest

- MAE: 22.96
- RMSE: 37.51
- MAPE: 5.67%

The model achieved a MAPE of 5.67%, meeting the project's target of 12%.

Congestion Classification

- Accuracy: 95.60%
- Macro F1-score: 0.9394

The model demonstrated good performance in classifying traffic congestion levels.

Accident Risk Prediction

- Actual ROC-AUC: 0.5779
- PRD Target ROC-AUC: 0.75

The accident risk model did not achieve the expected target and requires further improvement. The model is considered experimental.

7. Streamlit Dashboard Features

The FlowCast dashboard provides the following modules:

1. Dashboard Overview: Displays overall traffic statistics and key performance indicators.

2. Live Predictions: Displays predicted traffic volume, congestion, and accident risk estimates by road.

3. Traffic Analysis: Provides traffic volume analysis, travel-time analysis, road comparisons, and LSTM-based forecasting.

4. Congestion Analysis: Visualises congestion patterns, congestion distribution, and classification results.

5. Weather vs Traffic: Analyses the relationship between weather conditions and traffic patterns.

6. Accident Risk: Displays historical accident information and estimated accident risk.

7. Data Explorer: Provides dataset previews, missing-value analysis, data upload, and data quality checks.

8. Project Structure

The project is organised into the following main components:

- "data/" – Project datasets.
- "notebooks/" – Data analysis, preprocessing, and model development notebooks.
- "models/" – Saved machine learning and deep learning models.
- "src/" – Supporting Python source files.
- "flowcastapp.py" – Main Streamlit dashboard application.
- "requirements.txt" – Python package dependencies.
- "README.md" – Project documentation.

9. Installation and Setup

Step 1: Install Python

Install Python on your system.

Step 2: Open the Project Folder

Open the FlowCast project folder using VS Code or Anaconda Prompt.

Step 3: Install Dependencies

Run the following command:

pip install -r requirements.txt

Step 4: Run the Streamlit Dashboard

Execute the following command:

python -m streamlit run flowcastapp.py

Step 5: Access the Dashboard

The Streamlit application will open in your web browser.

10. Results and Outcomes

FlowCast demonstrates the use of Machine Learning and Deep Learning techniques for traffic analysis and prediction.

The Random Forest model achieved a MAPE of 5.67%, while the congestion classification model achieved an accuracy of 95.60%.

The LSTM model provides additional traffic forecasting functionality through the dashboard. The accident risk model requires further improvement to achieve the expected performance target.

The project provides an interactive platform for analysing traffic patterns and understanding predicted traffic conditions.

11. Future Enhancements

- Integration with real-time traffic sensor data.
- Improvement of accident risk prediction performance.
- Exploration of advanced forecasting techniques.
- Addition of interactive traffic maps and alerts.
- Cloud deployment for wider accessibility.

12. Author

Name: Yamini

Project Title: FlowCast – Smart Traffic Flow Prediction System

Project Domain: Artificial Intelligence and Machine Learning

Project Type: AI/ML Internship – Final Technical Project