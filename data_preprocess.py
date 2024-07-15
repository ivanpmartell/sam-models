import sys
import math
import numpy as np
from numpy.lib.stride_tricks import as_strided

sys.path.insert(1, sys.path[0])
from common import get_ss_q8, ss_index

def choose_preprocess(type:str) -> callable:
    """Converts string into function for preprocesses

    ### Parameters
    1. type : str
        - Type of preprocessing function as a string

    ### Returns
    - function
        - Preprocessing function matching the input string
    """
    if type == "onehot":
        return onehot_preprocess
    elif type == "nominal_location":
        return nominal_location_preprocess
    elif type == "nominal_circular":
        return nominal_circular_preprocess
    elif type == "frequency_location":
        return frequency_location_preprocess
    elif type == "frequency_circular":
        return frequency_circular_preprocess
    elif type == "frequency_max_location":
        return frequency_max_location_preprocess
    elif type == "frequency_max_circular":
        return frequency_max_circular_preprocess
    elif type == "nominal_windowed":
        return windowed_preprocess
    else:
        raise ValueError("Unknown preprocessing type given in command arguments")
    

def onehot_encode(Xy:np.ndarray) -> np.ndarray:
    """Encodes a single sequence into one-hot Q8 vectors

    ### Parameters
    1. Xy : ndarray
        - Input or target of a single sequence as given by the data script

    ### Returns
    - ndarray
        - n*m matrix, where n is the number of Q8 classes and m is the sequence length
    """
    classes = len(get_ss_q8())
    return np.eye(classes)[Xy].T

def onehot_preprocess(Xy:np.ndarray, max_len:int=1024) -> np.ndarray:
    """Encodes sequences into one-hot Q8 vectors

    ### Parameters
    1. Xy : ndarray
        - Input or target sequences as given by the data script
    2. max_len : int
        - Maximum sequence length expected

    ### Returns
    - ndarray
        - i*n*m*j matrix, where i = sequences, j = predictors n = Q8 classes, m = sequence length

    ### Notes
    When a single sequence is given, the returned matrix is n*m*j
    """
    len_predictors = Xy.shape[-1]//max_len
    classes = list(get_ss_q8())
    if len(Xy.shape) > 1:
        #TODO: make concated the right shape
        concated = np.zeros((len(Xy), len(classes), max_len))
        for i, prediction in enumerate(Xy):
            concated[i] = onehot_encode(prediction)
    else:
        concated = np.dstack(np.split(onehot_encode(Xy),len_predictors,axis=1))
    return concated

def frequency_preprocess(X:np.ndarray, max_len:int=1024) -> np.ndarray:
    """Encodes sequences into frequencies of Q8 for each position

    ### Parameters
    1. X : ndarray
        - Input sequences as given by the data script
    2. max_len : int
        - Maximum sequence length expected

    ### Returns
    - ndarray
        - TODO

    ### Notes
    If sequence exceeds max_len, sequence will be truncated.
    Alternatively if sequence is under max_len, sequence will be zero padded.
    """
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

def frequency_location_preprocess(X:np.ndarray, max_len:int=1024):
    """Performs frequency preprocessing with added location information at the end of each vector.

    ### Parameters
    1. X : ndarray
        - Input sequences as given by the data script
    2. max_len : int
        - Maximum sequence length expected

    ### Returns
    - ndarray
        - TODO

    ### Notes
    For use with inputs to match single target preprocessing.
    If sequence exceeds max_len, sequence will be truncated.
    Alternatively if sequence is under max_len, sequence will be zero padded.
    """
    len_classes = len(get_ss_q8())
    freqs = frequency_preprocess(X, max_len)
    freqs = freqs.swapaxes(1,2).reshape((len(X), max_len*len_classes))
    freqs = np.repeat(freqs, repeats=max_len, axis=0)
    location = np.tile(np.arange(1, max_len+1, dtype=np.int16), len(X))
    freqs = np.c_[freqs, location]
    return freqs

def frequency_circular_preprocess(X:np.ndarray, max_len:int=1024):
    """Performs frequency preprocessing by having the position of interest at the start of the vectors by circular shifting of the sequence.

    ### Parameters
    1. X : ndarray
        - Input sequences as given by the data script
    2. max_len : int
        - Maximum sequence length expected

    ### Returns
    - ndarray
        - TODO

    ### Notes
    For use with inputs to match single target preprocessing.
    If sequence exceeds max_len, sequence will be truncated.
    Alternatively if sequence is under max_len, sequence will be zero padded.
    """
    len_classes = len(get_ss_q8())
    freqs = frequency_preprocess(X, max_len)
    freqs = freqs.swapaxes(1,2).reshape((len(X), max_len*len_classes))
    freqs = np.repeat(freqs, repeats=max_len, axis=0)
    for i in range(len_classes):
        freqs[:, i*max_len:i*max_len+max_len] = custom_roll(freqs[:, i*max_len:i*max_len+max_len], max_len)
    return freqs

def frequency_max_preprocess(X:np.ndarray, max_len:int=1024):
    """Performs frequency preprocessing while returning only the most frequent ss for each position

    ### Parameters
    1. X : ndarray
        - Input sequences as given by the data script
    2. max_len : int
        - Maximum sequence length expected

    ### Returns
    - ndarray
        - TODO

    ### Notes
    For use with inputs to match single target preprocessing.
    If sequence exceeds max_len, sequence will be truncated.
    Alternatively if sequence is under max_len, sequence will be zero padded.
    """
    freqs = frequency_preprocess(X, max_len)
    freqs = np.argmax(freqs, axis=2)
    return freqs

def frequency_max_location_preprocess(X:np.ndarray, max_len:int=1024):
    """Performs maximum frequency preprocessing with added location information at the end of each vector.

    ### Parameters
    1. X : ndarray
        - Input sequences as given by the data script
    2. max_len : int
        - Maximum sequence length expected

    ### Returns
    - ndarray
        - TODO

    ### Notes
    For use with inputs to match single target preprocessing.
    If sequence exceeds max_len, sequence will be truncated.
    Alternatively if sequence is under max_len, sequence will be zero padded.
    """
    freqs = frequency_max_preprocess(X, max_len)
    freqs = np.repeat(freqs, repeats=max_len, axis=0)
    location = np.tile(np.arange(1, max_len+1, dtype=np.int16), len(X))
    freqs = np.c_[freqs, location]
    return freqs

def frequency_max_circular_preprocess(X:np.ndarray, max_len:int=1024):
    """Performs maximum frequency preprocessing by having the position of interest at the start of the vectors by circular shifting of the sequence.

    ### Parameters
    1. X : ndarray
        - Input sequences as given by the data script
    2. max_len : int
        - Maximum sequence length expected

    ### Returns
    - ndarray
        - TODO

    ### Notes
    For use with inputs to match single target preprocessing.
    If sequence exceeds max_len, sequence will be truncated.
    Alternatively if sequence is under max_len, sequence will be zero padded.
    """
    freqs = frequency_max_preprocess(X, max_len)
    freqs = np.repeat(freqs, repeats=max_len, axis=0)
    freqs = custom_roll(freqs, max_len)
    return freqs

def nominal_location_preprocess(X:np.ndarray, max_len:int=1024):
    """Adds location information at the end of each vector for nominal data.

    ### Parameters
    1. X : ndarray
        - Input sequences as given by the data script
    2. max_len : int
        - Maximum sequence length expected

    ### Returns
    - ndarray
        - TODO

    ### Notes
    For use with inputs to match single target preprocessing.
    If sequence exceeds max_len, sequence will be truncated.
    Alternatively if sequence is under max_len, sequence will be zero padded.
    """
    X_len = X.shape[0]
    len_predictors = X.shape[-1]//max_len
    X = np.repeat(X, repeats=max_len, axis=0)
    location = np.tile(np.arange(1, max_len+1, dtype=np.int16), X_len)
    X = np.c_[X, location]
    return X

def nominal_circular_preprocess(X:np.ndarray, max_len:int=1024):
    """Makes nominal data contain position of interest at the start of the vectors by circular shifting of the sequence.

    ### Parameters
    1. X : ndarray
        - Input sequences as given by the data script
    2. max_len : int
        - Maximum sequence length expected

    ### Returns
    - ndarray
        - TODO

    ### Notes
    For use with inputs to match single target preprocessing.
    If sequence exceeds max_len, sequence will be truncated.
    Alternatively if sequence is under max_len, sequence will be zero padded.
    """
    X_len = X.shape[0]
    len_predictors = X.shape[-1]//max_len
    X = np.repeat(X, repeats=max_len, axis=0)
    for i in range(len_predictors):
        X[:, i*max_len:i*max_len+max_len] = custom_roll(X[:, i*max_len:i*max_len+max_len], max_len)
    return X

def custom_roll(arr:np.ndarray, max_len:int):
    """Creates circular shifting vectors in a matrix. Each row shifts the position once

    ### Parameters
    1. arr : ndarray
        - Input array during preprocessing
    2. max_len : int
        - Maximum sequence length expected

    ### Returns
    - ndarray
        - TODO

    ### Notes
    For use with inputs to match single target preprocessing.
    """
    m = np.arange(max_len)*-1
    m = np.tile(m, len(arr)//max_len)
    arr_roll = arr[:, [*range(arr.shape[1]),*range(arr.shape[1]-1)]].copy()
    strd_0, strd_1 = arr_roll.strides
    n = arr.shape[1]
    result = as_strided(arr_roll, (*arr.shape, n), (strd_0 ,strd_1, strd_1))
    return result[np.arange(arr.shape[0]), (n-m)%n]

def single_target_preprocess(y:np.ndarray):
    """Outputs each value of a vector as a single entity.

    ### Parameters
    1. y : ndarray
        - Target sequences as given by the data script

    ### Returns
    - ndarray
        - TODO

    ### Notes
    For use with machine learning algorithms that can only output a single class at a time.
    """
    return y.ravel().astype(float)

def nominal_data(X:np.ndarray, numtype:type=float, max_len:int=1024):
    """Converts secondary structure sequences to nominal data

    ### Parameters
    1. X : ndarray
        - Input sequences as given by the data script
    2. numtype : type
        - Type of numerical values to utilize for memory purposes.
    3. max_len : int
        - Maximum sequence length expected

    ### Returns
    - ndarray
        - TODO

    ### Notes
    If sequence exceeds max_len, an error will occur.
    Alternatively if sequence is under max_len, sequence will be zero padded.
    """
    cats = np.zeros((max_len*len(X),), dtype=numtype)
    for i, prediction in enumerate(X):
        for j in range(len(prediction)):
            cats[max_len*i+j] = ss_index(prediction[j])
    return cats

def mutation_nominal_data(X:np.ndarray, mut_position:int, numtype:type=float, max_len:int=2048):
    """Converts secondary structure sequences to nominal data centered at mutation position by shifting the nominal vector

    ### Parameters
    1. X : ndarray
        - Input sequences as given by the data script
    2. mut_position : int
        - Position of the mutation in the sequence
    3. numtype : type
        - Type of numerical values to utilize for memory purposes.
    4. max_len : int
        - Maximum sequence length expected

    ### Returns
    - ndarray
        - TODO

    ### Notes
    If sequence exceeds max_len, an error will occur.
    Alternatively if sequence is under max_len, sequence will be zero padded.
    the midway point of the vector is the centered mutation location (e.g. 1024 for a maximum length of 2048)
    """
    cats = np.zeros((max_len*len(X),), dtype=numtype)
    for i, prediction in enumerate(X):
        for j in range(len(prediction)):
            cats[max_len*i+j] = ss_index(prediction[j])
        cats[i*max_len:i*max_len+max_len] = np.roll(cats[i*max_len:i*max_len+max_len], (max_len//2)-mut_position+1)
    return cats

# Window side length is used for each side of the current amino acid
#TODO: add preprocess type to windows
def windowed_preprocess(Xy, max_len, window_side_len=20, preprocess:str="nominal"):
    """Changes complete nominal data into a series of window sized nominal vectors.

    ### Parameters
    1. Xy : ndarray
        - Input sequences as given by the data script
    2. max_len : int
        - Maximum sequence length expected
    3. window_side_len : int
        - Length of each side of the window. The window length = window_side_len * 2 + 1

    ### Returns
    - ndarray
        - TODO

    ### Notes
    The windows are created by using the window_side_len parameter as the amount of positions to use on each side of the sequece.
    """
    Xy_len = Xy.shape[0]
    len_predictors = Xy.shape[-1]//max_len
    window_len = window_side_len * 2 + 1
    Xy = Xy.reshape((Xy_len, len_predictors, max_len))
    Xy = np.pad(Xy, ((0, 0), (0, 0), (window_side_len, window_side_len)))
    res = np.zeros((Xy_len * max_len, window_len * len_predictors))
    for i in range(max_len):
        if preprocess != "nominal":
            pp = choose_preprocess(preprocess)
            res[i::max_len] = pp(Xy[:,:,i:window_len+i]).reshape((Xy_len, -1))
        else:
            res[i::max_len] = Xy[:,:,i:window_len+i].reshape((Xy_len, -1))
    return res