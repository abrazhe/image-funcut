import numpy as np

def kalman_stack_filter(frames, seed='mean', gain=0.5, var=0.05, fn=lambda f:f):
    """Kalman stack filter similar to that of Imagej

    Input:
    ------
      - frames: array_like, e.g. nframes x nrows x ncolumns array or list of 2D
                images
      - seed: {'mean' | 'first' | 2D array}
              the seed to start with, defaults to the time-average of all
              frames, if 'first', then the first frame is used, if 2D array,
              this array is used as the seed
      - gain: overall filter gain
      - var: estimated environment noise variance

    Output:
    -------
      - new frames, an nframes x nrows x ncolumns array with filtered frames

    """
    if seed is 'mean' or None:
        seed = np.mean(frames,axis=0)
    elif seed is 'first':
        seed = frames[0]
    out = np.zeros_like(frames)
    predicted = fn(seed)
    Ek = var*np.ones_like(frames[0])
    var = Ek
    for k,M in enumerate(frames):
        Kk = 1.+ gain*(Ek/(Ek + var)-1)
        corrected = predicted + Kk*(M-predicted)
        err = (corrected-predicted)**2/predicted.max()**2 # unclear
        Ek = Ek*(1.-Kk) + err
        out[k] = corrected
        predicted = fn(out[k])
    return out

def test_kalman_stack_filter():
    "just tests if kalman_stack_filter function runs"
    print("Testing Kalman stack filter")
    try:
        test_arr = np.random.randn(100,64,64)
        arr2 = kalman_stack_filter(test_arr,)
        del  arr2, test_arr
    except Exception as e :
        print("filt.py: Failed to run kalman_stack_filter fuction")
        print("Reason:", e)
