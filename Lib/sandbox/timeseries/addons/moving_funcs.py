"""

A collection of moving functions for masked arrays and time series

:author: Pierre GF Gerard-Marchant & Matt Knox
:contact: pierregm_at_uga_dot_edu - mattknox_ca_at_hotmail_dot_com
:version: $Id: filters.py 2819 2007-03-03 23:00:20Z pierregm $
"""
__author__ = "Pierre GF Gerard-Marchant & Matt Knox ($Author: pierregm $)"
__version__ = '1.0'
__revision__ = "$Revision: 2819 $"
__date__     = '$Date: 2007-03-03 18:00:20 -0500 (Sat, 03 Mar 2007) $'

import numpy as N
from numpy import bool_, float_
narray = N.array

from scipy.signal import convolve, get_window

import maskedarray as MA
from maskedarray import MaskedArray, nomask, getmask, getmaskarray, masked
marray = MA.array

from timeseries.cseries import MA_mov_stddev, MA_mov_sum

__all__ = ['mov_sum',
           'mov_average_expw',
           'mov_stddev', 'mov_var', 'mov_sample_stddev', 'mov_sample_var',
           'cmov_average', 'cmov_mean', 'cmov_window'
           ]

def _process_result_dict(orig_data, result_dict):
    "process the results from the c function"

    rtype = result_dict['array'].dtype
    rmask = result_dict['mask']

    # makes a copy of the appropriate type
    data = orig_data.astype(rtype)
    data[:] = result_dict['array']

    return marray(data, mask=rmask, copy=True, subok=True)


def mov_sum(data, window_size, dtype=None):
    kwargs = {'array':data,
              'window_size':window_size}

    if dtype is not None:
        kwargs['dtype'] = dtype
              
    result_dict = MA_mov_sum(**kwargs)
    return _process_result_dict(data, result_dict)


def _mov_var_stddev(data, window_size, is_variance, is_sample, dtype):
    "helper function for mov_var and mov_stddev functions"

    kwargs = {'array':data,
              'window_size':window_size,
              'is_variance':is_variance,
              'is_sample':is_sample}
    
    if dtype is not None:
        kwargs['dtype'] = dtype
              
    result_dict = MA_mov_stddev(**kwargs)
    return _process_result_dict(data, result_dict)


def mov_var(data, window_size, dtype=None):
    """Calculates the moving variance of a 1-D array. This is the population
variance. See "mov_sample_var" for moving sample variance.

:Parameters:
    data : ndarray
        Data as a valid (subclass of) ndarray or MaskedArray. In particular, 
        TimeSeries objects are valid here.
    window_size : int 
        Time periods to use for each calculation.
    dtype : numpy data type specification (*None*)
        Behaves the same as the dtype parameter for the numpy.var function.
        
:Return value:
    The result is always a masked array (preserves subclass attributes). The
    result at index i uses values from [i-window_size:i+1], and will be masked
    for the first `window_size` values. The result will also be masked at i
    if any of the input values in the slice [i-window_size:i+1] are masked."""
        
    
    return _mov_var_stddev(data=data, window_size=window_size,
                           is_variance=1, is_sample=0, dtype=dtype)

def mov_stddev(data, window_size, dtype=None):
    """Calculates the moving standard deviation of a 1-D array. This is the
population standard deviation. See "mov_sample_stddev" for moving sample standard
deviation.

:Parameters:
    data : ndarray
        Data as a valid (subclass of) ndarray or MaskedArray. In particular, 
        TimeSeries objects are valid here.
    window_size : int 
        Time periods to use for each calculation.
    dtype : numpy data type specification (*None*)
        Behaves the same as the dtype parameter for the numpy.std function.
        
:Return value:
    The result is always a masked array (preserves subclass attributes). The
    result at index i uses values from [i-window_size:i+1], and will be masked
    for the first `window_size` values. The result will also be masked at i
    if any of the input values in the slice [i-window_size:i+1] are masked."""
    
    return _mov_var_stddev(data=data, window_size=window_size,
                           is_variance=0, is_sample=0, dtype=dtype)


def mov_sample_var(data, window_size, dtype=None):
    """Calculates the moving sample variance of a 1-D array.

:Parameters:
    data : ndarray
        Data as a valid (subclass of) ndarray or MaskedArray. In particular, 
        TimeSeries objects are valid here.
    window_size : int 
        Time periods to use for each calculation.
    dtype : numpy data type specification (*None*)
        Behaves the same as the dtype parameter for the numpy.var function.
        
:Return value:
    The result is always a masked array (preserves subclass attributes). The
    result at index i uses values from [i-window_size:i+1], and will be masked
    for the first `window_size` values. The result will also be masked at i
    if any of the input values in the slice [i-window_size:i+1] are masked."""
        
    
    return _mov_var_stddev(data=data, window_size=window_size,
                           is_variance=1, is_sample=1, dtype=dtype)

def mov_sample_stddev(data, window_size, dtype=None):
    """Calculates the moving sample standard deviation of a 1-D array.

:Parameters:
    data : ndarray
        Data as a valid (subclass of) ndarray or MaskedArray. In particular, 
        TimeSeries objects are valid here.
    window_size : int 
        Time periods to use for each calculation.
    dtype : numpy data type specification (*None*)
        Behaves the same as the dtype parameter for the numpy.std function.
        
:Return value:
    The result is always a masked array (preserves subclass attributes). The
    result at index i uses values from [i-window_size:i+1], and will be masked
    for the first `window_size` values. The result will also be masked at i
    if any of the input values in the slice [i-window_size:i+1] are masked."""
    
    return _mov_var_stddev(data=data, window_size=window_size,
                           is_variance=0, is_sample=1, dtype=dtype)

def mov_average_expw(data, span, tol=1e-6):
    """Calculates the exponentially weighted moving average of a series.

:Parameters:
    data : ndarray
        Data as a valid (subclass of) ndarray or MaskedArray. In particular, 
        TimeSeries objects are valid here.
    span : int 
        Time periods. The smoothing factor is 2/(span + 1)
    tol : float, *[1e-6]*
        Tolerance for the definition of the mask. When data contains masked 
        values, this parameter determinea what points in the result should be masked.
        Values in the result that would not be "significantly" impacted (as 
        determined by this parameter) by the masked values are left unmasked.
"""
    data = marray(data, copy=True, subok=True)
    ismasked = (data._mask is not nomask)
    data._mask = N.zeros(data.shape, bool_)
    _data = data._data
    #
    k = 2./float(span + 1)
    def expmave_sub(a, b):
        return b + k * (a - b)
    #
    data._data.flat = N.frompyfunc(expmave_sub, 2, 1).accumulate(_data)
    if ismasked:
        _unmasked = N.logical_not(data._mask).astype(float_)
        marker = 1. - N.frompyfunc(expmave_sub, 2, 1).accumulate(_unmasked)
        data._mask[marker > tol] = True
    data._mask[0] = True
    #
    return data

"""
def weightmave(data, span):
    data = marray(data, subok=True, copy=True)
    data._mask = N.zeros(data.shape, bool_)
    # Set the data
    _data = data._data
    tmp = N.empty_like(_data)
    tmp[:span] = _data[:span]
    s = 0
    for i in range(span, len(data)):
        s += _data[i] - _data[i-span]
        tmp[i] = span*_data[i] + tmp[i-1] - s
    tmp *= 2./(span*(n+1))
    data._data.flat = tmp
    # Set the mask
    if data._mask is not nomask:
        msk = data._mask.nonzero()[0].repeat(span).reshape(-1,span)
        msk += range(span)
        data._mask[msk.ravel()] = True
    data._mask[:span] = True
    return data
"""

#...............................................................................
def cmov_window(data, span, window_type):
    """Applies a centered moving window of type window_type and size span on the 
    data.
    
    Returns a (subclass of) MaskedArray. The k first and k last data are always 
    masked (with k=span//2). When data has a missing value at position i, 
    the result has missing values in the interval [i-k:i+k+1].
    
    
:Parameters:
    data : ndarray
        Data to process. The array should be at most 2D. On 2D arrays, the window
        is applied recursively on each column.
    span : integer
        The width of the window.
    window_type : string/tuple/float
        Window type (see Notes)
        
Notes
-----

The recognized window types are: boxcar, triang, blackman, hamming, hanning, 
bartlett, parzen, bohman, blackmanharris, nuttall, barthann, kaiser (needs beta), 
gaussian (needs std), general_gaussian (needs power, width), slepian (needs width).
If the window requires parameters, the window_type argument should be a tuple
with the first argument the string name of the window, and the next arguments 
the needed parameters. If window_type is a floating point number, it is interpreted 
as the beta parameter of the kaiser window.

Note also that only boxcar has been thoroughly tested.
    """
    #
    data = marray(data, copy=True, subok=True)
    if data._mask is nomask:
        data._mask = N.zeros(data.shape, bool_)
    window = get_window(window_type, span, fftbins=False)
    (n, k) = (len(data), span//2)
    #
    if data.ndim == 1:
        data._data.flat = convolve(data._data, window)[k:n+k] / float(span)
        data._mask[:] = ((convolve(getmaskarray(data), window) > 0)[k:n+k])
    elif data.ndim == 2:
        for i in range(data.shape[-1]):
            _data = data._data[:,i]
            _data.flat = convolve(_data, window)[k:n+k] / float(span)
            data._mask[:,i] = (convolve(data._mask[:,i], window) > 0)[k:n+k]
    else:
        raise ValueError, "Data should be at most 2D"
    data._mask[:k] = data._mask[-k:] = True
    return data

def cmov_average(data, span):
    """Computes the centered moving average of size span on the data.
    
    Returns a (subclass of) MaskedArray. The k first and k last data are always 
    masked (with k=span//2). When data has a missing value at position i, 
    the result has missing values in the interval [i-k:i+k+1].
    
:Parameters:
    data : ndarray
        Data to process. The array should be at most 2D. On 2D arrays, the window
        is applied recursively on each column.
    span : integer
        The width of the window.    
    """
    return cmov_window(data, span, 'boxcar')

cmov_mean = cmov_average
