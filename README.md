# Predict Automation First Break wave-P of Seismogram using LSTM 

This project was made as a campus final project, 
and it is hoped that it can be developed as further innovation. 
This application was created with the aim of determining the first time the 
P wave appears in seismogram data, so that later it will be useful for processing other data. 
This application is web-based by implementing the Streamlit framework and deployed to a 
public web server. 

Data processing stage:
- Seismogram data preprocessing (data normalization & filtering) 
- Create parameters consisting of STA/LTA, envelope, kurtosis, and skewness 
- Create a dataset consisting of 5 features and 1 label 
- Conduct dataset training with the Tensorflow deep learning model, until satisfactory accuracy and loss are obtained 
- Make data predictions on real data 
- Deployment to the web with Streamlit

After training a dataset of up to 50 epochs, with the bidirectional LSTM deep learning model, 
an accuracy of 98.5% and loss of up to 5% on training data is obtained, 
as well as an accuracy of 98.4% and a loss of up to 4.5% on data validation. 
After making predictions on real data, accurate results are obtained and in 
accordance with the predictions manually.
