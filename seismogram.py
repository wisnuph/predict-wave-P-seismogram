import pandas as pd
import numpy as np
from obspy.signal.trigger import classic_sta_lta, recursive_sta_lta
from obspy.signal.filter import envelope
from scipy.stats import kurtosis, skew


def runningKurt(x, N):
    y = np.zeros((len(x) - (N - 1),))
    for i in range(len(x) - (N - 1)):
         y[i] = kurtosis(x[i:(i + N)])
    return y

def runningSkew(x, N):
    y = np.zeros((len(x) - (N - 1),))
    for i in range(len(x) - (N - 1)):
         y[i] = skew(x[i:(i + N)])
    return y

def preprocess(stream, freq, corners):
  for i in range(len(stream)):
    stream[i] = stream[i].filter('lowpass', freq=freq, corners=corners, zerophase=True).normalize()
  return stream

def make_parameters(stream, sta, lta, sampling_rate):
  traces_classic_sta_lta = classic_sta_lta(stream[0], int(sta*sampling_rate), int(lta*sampling_rate))
  traces_recursive_sta_lta = recursive_sta_lta(stream[0], int(sta*sampling_rate), int(lta*sampling_rate))
  traces_envelope = envelope(stream[0].data)
  traces_kurtosis = list(runningKurt(stream[0].data, 360))
  traces_kurtosis[:0] = [0 for i in range(len(traces_kurtosis), len(stream[0].data), 1)]
  traces_skewness = list(runningSkew(stream[0].data, 360))
  traces_skewness[:0] = [0 for i in range(len(traces_skewness), len(stream[0].data), 1)]

  data_example = {'classic STA/LTA': traces_classic_sta_lta,
                  'recursive STA/LTA': traces_recursive_sta_lta,
                  'envelope': traces_envelope,
                  'kurtosis': traces_kurtosis,
                  'skewness': traces_skewness
                  }
  df_example = pd.DataFrame(data_example)

  return df_example

