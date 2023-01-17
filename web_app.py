import streamlit as st
from obspy import read
import matplotlib.pyplot as plt
import seismogram
import tensorflow as tf
import numpy as np


def main():
    st.set_page_config(page_title="Predict P Wave", layout="wide")
    page_settings = """
    <style>
    
    #root > div:nth-child(1) > div > div > div > div > section > div {
    padding-top: 4rem;
    }

    [data-testid="stHeader"]{
    background-color: rgb(173,216,230);
    }

    .stApp {
    background-image: url();
    background-attachment: fixed;
    background-size: cover;
    }
    </style>
    """
    st.markdown(page_settings, unsafe_allow_html=True)
    st.title("Application of Predict First Break Wave-P on Seismogram")
    st.markdown("**This is a web to predict first break wave-P using RNN (Recurrent Neural Network)**")
    st.markdown("[Get data seismogram](https://ds.iris.edu/wilber3/find_event)")
    st.markdown("---")
    
    trace_file = st.file_uploader("Upload Data Seismogram", type=['miniseed', 'mseed'])
    savedModel = tf.keras.models.load_model('model_lstm.h5')

    if trace_file is not None:
        st.header("Example of Trace Seismogram")
        trace = read(trace_file)
        st.pyplot(trace[0].plot())
       
        if st.button('Predict'):
            trace_n = trace.normalize()
            df = seismogram.make_parameters(trace_n, 2, 30, 40)
            df = np.resize(df, (df.shape[0], 1, df.shape[1]))
            predict = savedModel.predict(df)
            predict = np.resize(predict, (predict.shape[1], predict.shape[0]))
            idx_predict = predict[0].argmax()

            fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(15,10))
            ax1.plot(predict[0], label='Machine Learning Output', color='black')
            ax1.legend()
            ax2.plot(trace_n[0].data, label='Trace Example', color='black')
            ax2.axvline(x=idx_predict, linewidth=1, label='First break', color='red')
            ax2.legend()
            st.pyplot(fig1)
            
            st.markdown('*Note: The arrival of the wave-P is determined based on the highest value of the machine learning output*')
            
        if st.button('Predict with filterisasi Trace'):
            trace_n = preprocess(trace_n, 1, 3)
            df = seismogram.make_parameters(trace_n, 2, 30, 40)
            df = np.resize(df, (df.shape[0], 1, df.shape[1]))
            predict = savedModel.predict(df)
            predict = np.resize(predict, (predict.shape[1], predict.shape[0]))
            idx_predict = predict[0].argmax()

            fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(15,10))
            ax1.plot(predict[0], label='Machine Learning Output', color='black')
            ax1.legend()
            ax2.plot(trace_n[0].data, label='Trace Example', color='black')
            ax2.axvline(x=idx_predict, linewidth=1, label='First break', color='red')
            ax2.legend()
            st.pyplot(fig2)

            st.markdown('*Note: The arrival of the wave-P is determined based on the highest value of the machine learning output*')

if __name__ == "__main__":
    main()
