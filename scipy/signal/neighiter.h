#ifndef _SCIPY_SIGNAL_NEIGHITER_H_
#define _SCIPY_SIGNAL_NEIGHITER_H_

#include <Python.h>

#define PY_ARRAY_UNIQUE_SYMBOL _scipy_signal_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/ndarrayobject.h>

typedef struct {
    PyObject_HEAD

    /* PyArrayIterObject part: keep this in this exact order */
    int               nd_m1;            /* number of dimensions - 1 */
    npy_intp          index, size;
    npy_intp          coordinates[NPY_MAXDIMS];/* N-dimensional loop */
    npy_intp          dims_m1[NPY_MAXDIMS];    /* ao->dimensions - 1 */
    npy_intp          strides[NPY_MAXDIMS];    /* ao->strides or fake */
    npy_intp          backstrides[NPY_MAXDIMS];/* how far to jump back */
    npy_intp          factors[NPY_MAXDIMS];     /* shape factors */
    PyArrayObject     *ao;
    char              *dataptr;        /* pointer to current item*/
    npy_bool          contiguous;

    /* New members */
    npy_intp nd;

    /* Dimensions is the dimension of the array */
    npy_intp dimensions[NPY_MAXDIMS];
    /* Bounds of the neighborhood to iterate over */
    npy_intp bounds[NPY_MAXDIMS][2];

    /* Neighborhood points coordinates are computed relatively to the point pointed
     * by _internal_iter */
    PyArrayIterObject* _internal_iter;
    /* To keep a reference to the zero representation correponding to the dtype
     * of the array we iterate over */
    char* zero;
} PyArrayNeighIterObject;

PyTypeObject PyArrayNeighIter_Type;

/*
 * Public API
 */

/* 
 * Main ctor
 *
 * bounds is expected to be a (2 * iter->ao->nd) arrays, such as the range
 * bound[2*i]->bounds[2*i+1] defines the range where to walk for dimension i
 * (both bounds are included in the walked coordinates). 
 *
 * Example:
 *      PyArrayIterObject *iter, *neigh_iter_base;
 *      PyArrayNeighIterObject *neigh_iter;
 *      iter = PyArray_IterNew(x);
 *       
 *      // For a 3x3 kernel
 *      bounds = {-1, 1, -1, 1};
 *      neigh_iter = PyArrayNeighIter_New(iter, bounds);
 *      // Hack so that neigh_iter_base points to iter, but giving access to
 *      // the base properties of iter (PyArrayIterObject is inherited from
 *      // PyArrayObject)
 *      neigh_iter_base = (PyArrayIterObject*)iter;
 *
 *      for(i = 0; i < iter->size; ++i) {
 *              for (j = 0; j < neigh_iter_base->size; ++j) {
 *                      // Walk around the item currently pointed by iter->dataptr
 *                      PyArrayNeighIter_Next(neigh_iter);
 *              }
 *
 *              // Move to the next point of iter
 *              PyArrayIter_Next(iter);
 *              PyArrayNeighIter_Reset(neigh_iter);
 *      }
 *
 * If the coordinates point to a point outside the array, the iterator points
 * to 0. More elaborate schemes (constants, mirroring, repeat) may be
 * implemented later.
 */
PyArrayNeighIterObject*
PyArrayNeighIter_New(PyArrayIterObject* iter, const npy_intp *bounds);

static NPY_INLINE int PyArrayNeighIter_Reset(PyArrayNeighIterObject* iter);
static NPY_INLINE int PyArrayNeighIter_Next(PyArrayNeighIterObject* iter);

/*
 * Private API (here for inline)
 */
static NPY_INLINE int _PyArrayNeighIter_IncrCoord(PyArrayNeighIterObject* iter);
static NPY_INLINE int _PyArrayNeighIter_SetPtr(PyArrayNeighIterObject* iter);

/*
 * Inline implementations
 */
static NPY_INLINE int PyArrayNeighIter_Reset(PyArrayNeighIterObject* iter)
{
    int i;

    for(i = 0; i < iter->nd; ++i) {
        iter->coordinates[i] = iter->bounds[i][0];
    }
    _PyArrayNeighIter_SetPtr(iter);

    return 0;
}

/*
 * Update to next item of the iterator
 *
 * Note: this simply increment the coordinates vector, last dimension
 * incremented first , i.e, for dimension 3
 * ...
 * -1, -1, -1
 * -1, -1,  0
 * -1, -1,  1
 *  ....
 * -1,  0, -1
 * -1,  0,  0
 *  ....
 * 0,  -1, -1
 * 0,  -1,  0
 *  ....
 */
static NPY_INLINE int _PyArrayNeighIter_IncrCoord(PyArrayNeighIterObject* iter)
{
    int i, wb;

#define _update_coord_iter(c) \
    wb = iter->coordinates[c] < iter->bounds[c][1]; \
    if (wb) { \
        iter->coordinates[c] += 1; \
        return 0; \
    } \
    else { \
        iter->coordinates[c] = iter->bounds[c][0]; \
    }

    for(i = iter->nd-1; i >= 0; --i) {
        _update_coord_iter(i)
    }

    return 0;
}

static NPY_INLINE int _PyArrayNeighIter_IncrCoord2D(PyArrayNeighIterObject* iter)
{
    int wb;

    _update_coord_iter(1)
    _update_coord_iter(0)

    return 0;
}
#undef _update_coord_iter

/* set the dataptr from its current coordinates */
static NPY_INLINE int _PyArrayNeighIter_SetPtr(PyArrayNeighIterObject* iter)
{
    int i;
    npy_intp offset, bd;

    iter->dataptr = iter->_internal_iter->dataptr;

#define _inc_set_ptr(c) \
    bd = iter->coordinates[c] + iter->_internal_iter->coordinates[c]; \
    if (bd < 0 || bd > iter->dimensions[c]) { \
        iter->dataptr = iter->zero; \
        return 1; \
    } \
    offset = iter->coordinates[c] * iter->strides[c]; \
    iter->dataptr += offset;

    for(i = 0; i < iter->nd; ++i) {
        _inc_set_ptr(i)
    }

    return 0;
}

static NPY_INLINE int _PyArrayNeighIter_SetPtr2D(PyArrayNeighIterObject* iter)
{
    npy_intp offset, bd;

    iter->dataptr = iter->_internal_iter->dataptr;

    _inc_set_ptr(0)
    _inc_set_ptr(1)

    return 0;
}
#undef _inc_set_ptr

/*
 * Advance to the next neighbour
 */
static NPY_INLINE int PyArrayNeighIter_Next2D(PyArrayNeighIterObject* iter)
{
    _PyArrayNeighIter_IncrCoord2D(iter);
    _PyArrayNeighIter_SetPtr2D(iter);

    return 0;
}

static NPY_INLINE int PyArrayNeighIter_Next(PyArrayNeighIterObject* iter)
{
    _PyArrayNeighIter_IncrCoord (iter);
    _PyArrayNeighIter_SetPtr(iter);

    return 0;
}
#endif
