import sys
import math
import numpy as np
from numpy.lib.stride_tricks import as_strided

sys.path.insert(1, sys.path[0])
from common import get_ss_q8, ss_index

def choose_preprocess(type):
    if type == "single_target":
        return single_target_preprocess
    if type == "nominal_location":
        return nominal_location_preprocess
    elif type == "onehot":
        return onehot_preprocess
    elif type == "frequency":
        return frequency_preprocess
    elif type == "frequency_location":
        return frequency_location_preprocess
    elif type == "frequency_max_location":
        return frequency_max_location_preprocess
    elif type == "nominal_windowed":
        return nominal_windowed_preprocess
    else:
        raise ValueError("Unknown preprocessing type given in command arguments")
    

def onehot_encode(X):
    classes = len(get_ss_q8())
    return np.eye(classes)[X]

def onehot_preprocess(Xy, max_len=1024, windowed=False):
    len_predictors = Xy.shape[-1]//max_len
    classes = list(get_ss_q8())
    if len(Xy.shape) > 1:
        concated = np.zeros((len(Xy), max_len* len_predictors, len(classes)))
        for i, prediction in enumerate(Xy):
            concated[i] = onehot_encode(prediction)
        concated = concated.reshape(len(Xy), len_predictors, max_len, len(classes))
    else:
        concated = onehot_encode(Xy).reshape(len_predictors, max_len, len(classes))
    return concated

def frequency_preprocess(X, max_len=1024, windowed=False):
    len_predictors = X.shape[-1]//max_len
    len_classes = len(get_ss_q8())
    freqs = np.zeros((len(X), max_len, len_classes))
    for i in range(len(X)):
        for j in range(max_len):
            for k in range(len_predictors):
                freqs[i, j, X[i,max_len*k+j]] += 1
    max_class_freqs = np.full((len(X), max_len, 1), 9, dtype=np.int8)
    freqs = np.divide(freqs, max_class_freqs, out=np.zeros_like(freqs), where=max_class_freqs!=0)
    return freqs

def frequency_location_preprocess(X, max_len=1024, circular=False):
    len_classes = len(get_ss_q8())
    freqs = frequency_preprocess(X, max_len, False)
    freqs = freqs.swapaxes(1,2).reshape((len(X), max_len*len_classes))
    freqs = np.repeat(freqs, repeats=max_len, axis=0)
    if circular:
        for i in range(len_classes):
            freqs[:, i*max_len:i*max_len+max_len] = custom_roll(freqs[:, i*max_len:i*max_len+max_len], max_len)
    else:
        location = np.tile(np.arange(1, max_len+1, dtype=np.int16), len(X))
        freqs = np.c_[freqs, location]
    return freqs

def frequency_circular_location_preprocess(X, max_len=1024):
    return frequency_location_preprocess(X, max_len, True)

def frequency_max_preprocess(X, max_len=1024, windowed=False):
    len_predictors = X.shape[-1]//max_len
    len_classes = len(get_ss_q8())
    freqs = np.zeros((len(X), max_len, len_classes))
    for i in range(len(X)):
        for j in range(max_len):
            for k in range(len_predictors):
                freqs[i, j, X[i,max_len*k+j]] += 1
    freqs = np.argmax(freqs,axis=2)
    return freqs

def frequency_max_location_preprocess(X, max_len=1024, circular=False):
    len_predictors = X.shape[-1]//max_len
    len_classes = len(get_ss_q8())
    freqs = np.zeros((len(X), max_len, len_classes))
    for i in range(len(X)):
        for j in range(max_len):
            for k in range(len_predictors):
                freqs[i, j, X[i,max_len*k+j]] += 1
    freqs = np.argmax(freqs,axis=2)
    freqs = np.repeat(freqs, repeats=max_len, axis=0)
    if circular:
        freqs = custom_roll(freqs, max_len)
    else:
        location = np.tile(np.arange(1, max_len+1, dtype=np.int16), len(X))
        freqs = np.c_[freqs, location]
    return freqs

def frequency_circular_max_location_preprocess(X, max_len=1024):
    return frequency_max_location_preprocess(X, max_len, True)

def nominal_location_preprocess(X, max_len=1024, circular=False):
    X_len = X.shape[0]
    len_predictors = X.shape[-1]//max_len
    X = np.repeat(X, repeats=max_len, axis=0)
    if circular:
        for i in range(len_predictors):
            X[:, i*max_len:i*max_len+max_len] = custom_roll(X[:, i*max_len:i*max_len+max_len], max_len)
    else:
        location = np.tile(np.arange(1, max_len+1, dtype=np.int16), X_len)
        X = np.c_[X, location]
    return X

def nominal_circular_location_preprocess(X, max_len=1024):
    return nominal_location_preprocess(X, max_len, True)

def custom_roll(arr, max_len=1024):
    m = np.arange(max_len)*-1
    m = np.tile(m, len(arr)//max_len)
    arr_roll = arr[:, [*range(arr.shape[1]),*range(arr.shape[1]-1)]].copy()
    strd_0, strd_1 = arr_roll.strides
    n = arr.shape[1]
    result = as_strided(arr_roll, (*arr.shape, n), (strd_0 ,strd_1, strd_1))
    return result[np.arange(arr.shape[0]), (n-m)%n]

def single_target_preprocess(y):
    return y.ravel().astype(float)

def nominal_data(X, numtype=float, max_len=1024):
    cats = np.zeros((max_len*len(X),), dtype=numtype)
    for i, prediction in enumerate(X):
        for j in range(len(prediction)):
            cats[max_len*i+j] = ss_index(prediction[j])
    return cats

def mutation_nominal_data(X, mut_position, numtype=float, max_len=2048):
    cats = np.zeros((max_len*len(X),), dtype=numtype)
    for i, prediction in enumerate(X):
        for j in range(len(prediction)):
            cats[max_len*i+j] = ss_index(prediction[j])
        cats[i*max_len:i*max_len+max_len] = np.roll(cats[i*max_len:i*max_len+max_len], (max_len//2)-mut_position+1) #midway point is the centered mutation location (e.g. 1024 for 2048 max len)
    return cats

# Window side length is used for each side of the current amino acid
def nominal_windowed_preprocess(Xy, max_len, max_window_side_len=20):
    Xy_len = Xy.shape[0]
    len_predictors = Xy.shape[-1]//max_len
    window_len = max_window_side_len * 2 + 1
    Xy = Xy.reshape((Xy_len, len_predictors, max_len))
    Xy = np.pad(Xy, ((0, 0), (0, 0), (max_window_side_len, max_window_side_len)))
    res = np.zeros((Xy_len * max_len, window_len * len_predictors))
    for i in range(max_len):
        res[i::max_len] = Xy[:,:,i:window_len+i].reshape((Xy_len, -1))
    return res

def frequency_windowed_preprocess(Xy, max_window_side_len=20, max_len=1024):
    window_len = max_window_side_len * 2 + 1
    return Xy.reshape(math.ceil(max_len/window_len), window_len)

def onehot_windowed_preprocess(Xy, max_window_side_len=20, max_len=1024):
    window_len = max_window_side_len * 2 + 1
    return Xy.reshape(math.ceil(max_len/window_len), window_len)