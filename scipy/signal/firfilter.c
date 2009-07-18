#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL _scipy_signal_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/noprefix.h>

#include "sigtools.h"
#define PYERR(message) {PyErr_SetString(PyExc_ValueError, message);}
static int elsizes[] = {sizeof(Bool),
						sizeof(byte),
                        sizeof(ubyte),
                        sizeof(short),
                        sizeof(ushort),
                        sizeof(int),
						sizeof(uint),
						sizeof(long),
                        sizeof(ulong),
                        sizeof(longlong),
						sizeof(ulonglong),
                        sizeof(float),
                        sizeof(double),
						sizeof(longdouble),
                        sizeof(cfloat),
                        sizeof(cdouble),
						sizeof(clongdouble),
                        sizeof(void *),
						0,0,0,0};

typedef void (OneMultAddFunction) (char *, char *, char *);

#define MAKE_ONEMULTADD(fname, type) \
static void fname ## _onemultadd(char *sum, char *term1, char *term2) { \
  (*((type *) sum)) += (*((type *) term1)) * \
  (*((type *) term2)); return; }

MAKE_ONEMULTADD(UBYTE, ubyte)
MAKE_ONEMULTADD(USHORT, ushort)
MAKE_ONEMULTADD(UINT, uint)
MAKE_ONEMULTADD(ULONG, ulong)
MAKE_ONEMULTADD(ULONGLONG, ulonglong)

MAKE_ONEMULTADD(BYTE, byte)
MAKE_ONEMULTADD(SHORT, short)
MAKE_ONEMULTADD(INT, int)
MAKE_ONEMULTADD(LONG, long)
MAKE_ONEMULTADD(LONGLONG, longlong)

MAKE_ONEMULTADD(FLOAT, float)
MAKE_ONEMULTADD(DOUBLE, double)
MAKE_ONEMULTADD(LONGDOUBLE, longdouble)
 
#ifdef __GNUC__
MAKE_ONEMULTADD(CFLOAT, __complex__ float)
MAKE_ONEMULTADD(CDOUBLE, __complex__ double)
MAKE_ONEMULTADD(CLONGDOUBLE, __complex__ long double)
#else
#define MAKE_C_ONEMULTADD(fname, type) \
static void fname ## _onemultadd(char *sum, char *term1, char *term2) { \
  ((type *) sum)[0] += ((type *) term1)[0] * ((type *) term2)[0] \
    - ((type *) term1)[1] * ((type *) term2)[1]; \
  ((type *) sum)[1] += ((type *) term1)[0] * ((type *) term2)[1] \
    + ((type *) term1)[1] * ((type *) term2)[0]; \
  return; }
MAKE_C_ONEMULTADD(CFLOAT, float)
MAKE_C_ONEMULTADD(CDOUBLE, double)
MAKE_C_ONEMULTADD(CLONGDOUBLE, longdouble)
#endif /* __GNUC__ */

static OneMultAddFunction *OneMultAdd[]={NULL,
					 					 BYTE_onemultadd,
					 					 UBYTE_onemultadd,
					 					 SHORT_onemultadd,
                                         USHORT_onemultadd,
					 					 INT_onemultadd,
                                         UINT_onemultadd,
					 					 LONG_onemultadd,
					 					 ULONG_onemultadd,
					 					 LONGLONG_onemultadd,
					 					 ULONGLONG_onemultadd,
					 					 FLOAT_onemultadd,
					 					 DOUBLE_onemultadd,
					 					 LONGDOUBLE_onemultadd,
					 					 CFLOAT_onemultadd,
					 					 CDOUBLE_onemultadd,
					 					 CLONGDOUBLE_onemultadd,
                                         NULL, NULL, NULL, NULL};

void convolve2d_worker(PyArrayIterObject *itSignal, 
			 		   PyArrayNeighborhoodIterObject *curSignal,  
			 		   PyArrayNeighborhoodIterObject *curKern, 
			 		   PyArrayIterObject *itOut, 
			 		   int typenum)
  {//only need what's there.  all the boundaries and whatnot are handled by sigtools.
  OneMultAddFunction *mult_and_add;
  int i, j;
  mult_and_add = OneMultAdd[typenum];
  if (mult_and_add == NULL) {PYERR("Convolve not available for this type");}
  int *zero = 0;
  
  PyArray_ITER_RESET(itOut);
  
  for (i = 0;i<itOut->size;++i){
  	mult_and_add((char *)itOut->dataptr,(char *)zero,(char *)zero);
    PyArrayNeighborhoodIter_Reset(curSignal);
    PyArrayNeighborhoodIter_Reset(curKern);
  	for (j = 0; j < curKern->size;++j){

  		
  		mult_and_add((char *)itOut->dataptr,(char *)curSignal->dataptr,(char *)curKern->dataptr);
  		
  		PyArrayNeighborhoodIter_Next(curSignal);
  		PyArrayNeighborhoodIter_Next(curKern);
	}
	PyArray_ITER_NEXT(itSignal);
	PyArray_ITER_NEXT(itOut);

}
  return ;
}


