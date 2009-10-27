import numpy as np
cimport numpy as np

# XXX: fix indexing, do not use int
ctypedef int index_t

cdef extern void dia_matvec_int_npy_int8(index_t n_row, index_t n_col,
        index_t n_diags,
        index_t L, index_t *offsets, np.npy_int8 diags[], np.npy_int8 Xx[],
        np.npy_int8 Yx[])

cdef extern void dia_matvec_int_npy_uint8(index_t n_row, index_t n_col,
        index_t n_diags,
        index_t L, index_t *offsets, np.npy_uint8 diags[], np.npy_uint8 Xx[],
        np.npy_uint8 Yx[])

cdef extern void dia_matvec_int_npy_int16(index_t n_row, index_t n_col,
        index_t n_diags,
        index_t L, index_t *offsets, np.npy_int16 diags[], np.npy_int16 Xx[],
        np.npy_int16 Yx[])

cdef extern void dia_matvec_int_npy_uint16(index_t n_row, index_t n_col,
        index_t n_diags,
        index_t L, index_t *offsets, np.npy_uint16 diags[], np.npy_uint16 Xx[],
        np.npy_uint16 Yx[])

cdef extern void dia_matvec_int_npy_int32(index_t n_row, index_t n_col,
        index_t n_diags,
        index_t L, index_t *offsets, np.npy_int32 diags[], np.npy_int32 Xx[],
        np.npy_int32 Yx[])

cdef extern void dia_matvec_int_npy_uint32(index_t n_row, index_t n_col,
        index_t n_diags,
        index_t L, index_t *offsets, np.npy_uint32 diags[], np.npy_uint32 Xx[],
        np.npy_uint32 Yx[])

cdef extern void dia_matvec_int_npy_int64(index_t n_row, index_t n_col,
        index_t n_diags,
        index_t L, index_t *offsets, np.npy_int64 diags[], np.npy_int64 Xx[],
        np.npy_int64 Yx[])

cdef extern void dia_matvec_int_npy_uint64(index_t n_row, index_t n_col,
        index_t n_diags,
        index_t L, index_t *offsets, np.npy_uint64 diags[], np.npy_uint64 Xx[],
        np.npy_uint64 Yx[])

cdef extern void dia_matvec_int_double(index_t n_row, index_t n_col,
        index_t n_diags,
        index_t L, index_t *offsets, double diags[], double Xx[], double Yx[])

cdef extern void dia_matvec_int_float(index_t n_row, index_t n_col,
        index_t n_diags,
        index_t L, index_t *offsets, float diags[], float Xx[], float Yx[])

cdef extern void dia_matvec_int_npy_longdouble(index_t n_row, index_t n_col,
        index_t n_diags,
        index_t L, index_t *offsets,
        np.npy_longdouble diags[], np.npy_longdouble Xx[], np.npy_longdouble Yx[])

def dia_matvec(index_t n_row, index_t n_col, index_t n_diags, 
               index_t L, np.ndarray offsets, np.ndarray diags,
               np.ndarray Xx, np.ndarray Yx):
    cdef np.ndarray safe_diags, safe_Xx, safe_Yx, safe_offsets

    if not offsets.dtype == np.int:
        raise ValueError("Expected %s for offsets, got %s" % 
                         (np.int, offsets.dtype))
    safe_offsets = np.ascontiguousarray(offsets, np.int)

    t = None
    for _t in [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32,
               np.int64, np.uint64, np.float32, np.float64, np.longdouble]:
        if np.can_cast(diags.dtype, t) and np.can_cast(Xx.dtype, t) \
                and np.can_cast(Yx.dtype, t):
            t = _t
            break

    if t is None:
        raise ValueError("type not supported %s - %s - %s" % \
                         (diags.dtype, Xx.dtype, Yx.dtype))

    safe_diags = np.ascontiguousarray(diags, t)
    safe_Xx = np.ascontiguousarray(Xx, t)
    safe_Yx = np.ascontiguousarray(Yx, t)
    if t == np.int8:
        dia_matvec_int_npy_int8(n_row, n_col, n_diags, L, <index_t*>safe_offsets.data,
                              <np.npy_int8*>safe_diags.data,
                              <np.npy_int8*>safe_Xx.data,
                              <np.npy_int8*>safe_Yx.data)
    elif t == np.uint8:
        dia_matvec_int_npy_uint8(n_row, n_col, n_diags, L, <index_t*>safe_offsets.data,
                              <np.npy_uint8*>safe_diags.data,
                              <np.npy_uint8*>safe_Xx.data,
                              <np.npy_uint8*>safe_Yx.data)
    elif t == np.int16:
        dia_matvec_int_npy_int16(n_row, n_col, n_diags, L, <index_t*>safe_offsets.data,
                              <np.npy_int16*>safe_diags.data,
                              <np.npy_int16*>safe_Xx.data,
                              <np.npy_int16*>safe_Yx.data)
    elif t == np.uint16:
        dia_matvec_int_npy_uint16(n_row, n_col, n_diags, L, <index_t*>safe_offsets.data,
                              <np.npy_uint16*>safe_diags.data,
                              <np.npy_uint16*>safe_Xx.data,
                              <np.npy_uint16*>safe_Yx.data)
    elif t == np.int32:
        dia_matvec_int_npy_int32(n_row, n_col, n_diags, L, <index_t*>safe_offsets.data,
                              <np.npy_int32*>safe_diags.data,
                              <np.npy_int32*>safe_Xx.data,
                              <np.npy_int32*>safe_Yx.data)
    elif t == np.uint32:
        dia_matvec_int_npy_uint32(n_row, n_col, n_diags, L, <index_t*>safe_offsets.data,
                              <np.npy_uint32*>safe_diags.data,
                              <np.npy_uint32*>safe_Xx.data,
                              <np.npy_uint32*>safe_Yx.data)
    elif t == np.int64:
        dia_matvec_int_npy_int64(n_row, n_col, n_diags, L, <index_t*>safe_offsets.data,
                              <np.npy_int64*>safe_diags.data,
                              <np.npy_int64*>safe_Xx.data,
                              <np.npy_int64*>safe_Yx.data)
    elif t == np.uint64:
        dia_matvec_int_npy_uint64(n_row, n_col, n_diags, L, <index_t*>safe_offsets.data,
                              <np.npy_uint64*>safe_diags.data,
                              <np.npy_uint64*>safe_Xx.data,
                              <np.npy_uint64*>safe_Yx.data)
    elif t == np.float32:
        dia_matvec_int_float(n_row, n_col, n_diags, L, <index_t*>safe_offsets.data,
                              <float*>safe_diags.data,
                              <float*>safe_Xx.data,
                              <float*>safe_Yx.data)
    elif t == np.float64:
        dia_matvec_int_double(n_row, n_col, n_diags, L, <index_t*>safe_offsets.data,
                              <double*>safe_diags.data,
                              <double*>safe_Xx.data,
                              <double*>safe_Yx.data)
    elif t == np.longdouble:
        dia_matvec_int_npy_longdouble(n_row, n_col, n_diags, L,
                              <index_t*>safe_offsets.data,
                              <np.npy_longdouble*>safe_diags.data,
                              <np.npy_longdouble*>safe_Xx.data,
                              <np.npy_longdouble*>safe_Yx.data)
    else:
        raise ValueError("Type %s not supported yet" % t)

    return safe_Yx
