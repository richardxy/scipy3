"""
Implementation of Harwell-Boeing read/write.
"""
# Really crude at the moment:
#   - only support for assembled, non-symmetric, real matrices
#   - only support integer for pointer/indices
#   - only support exponential format for values

# TODO:
#   - support for more value types (other float format at least)
#   - API to get/write header (title, key, etc...) ?
#   - API for format specifier in hb_write ?
#   - Add more support (symmetric/complex matrices, non-assembled matrices ?)

# XXX: read_hb is reasonably efficient (>= 85 % is in numpy.fromstring), being
# faster would require compiled code. Although not a terribly exciting task,
# having fast, reusable facilities to read fortran-formatted files would be
# useful outside this module.
import numpy as np

from scipy.sparse \
    import \
        csc_matrix
from scipy.sparse.io._fortran_format_parser \
    import\
        FortranFormatParser, IntFormat, ExpFormat

__all__ = ["MalformedHeader", "read_hb", "write_hb"]

class MalformedHeader(Exception):
    pass

class HarwellBoeingHeader(object):
    @classmethod
    def from_file(cls, fid):
        return _read_header(fid)

    def __init__(self, title,
                 pointer_nlines, pointer_nitems, pointer_width,
                 indices_nlines, indices_nitems, indices_width,
                 values_nlines, values_nitems, values_width,
                 n_rows, n_cols, n_nzeros,
                 key=None):
        self.title = title
        self.key = key

        self.pointer_nlines = pointer_nlines
        self.pointer_nitems = pointer_nitems
        self.pointer_width = pointer_width

        self.indices_nlines = indices_nlines
        self.indices_nitems = indices_nitems
        self.indices_width = indices_width

        self.values_nlines = values_nlines
        self.values_nitems = values_nitems
        self.values_width = values_width

        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_nzeros = n_nzeros

def _expect_int(value, msg=None):
    try:
        return int(value)
    except ValueError:
        if msg is None:
            msg = "Expected an int, got %s"
        raise ValueError(msg % value)

def fortran_format(n, prec=None):
    if n.dtype in (np.int, np.int32):
        if prec is not None:
            raise ValueError(
                    "prec argument does not make sense for integer value")
        ndigits = np.floor(np.log10(n)) + 1
        nvals = 80 / (ndigits + 1)
        return "(%dI%d)" % (nvals, ndigits + 1)
    elif n.dtype == np.float:
        # Only exponential format for now
        if prec is None:
            prec = 8
        # Number of digits in the exponent
        exp = np.floor(np.log10(n)) + 1
        ndigits = np.max([2, number_digits(exp)])

        # len of one number: sign + 0 + "." +
        # number of digit for fractional part + 'E' + sign of exponent +
        # len of exponent
        nval_sz = 1 + 1 + 1 + prec + 1 + 1 + ndigits
        nvals = 80 /(nval_sz + 1)
        return "(%dE%d.%d)" % (nvals, nval_sz + 1, prec)
    else:
        raise ValueError("unsupported type %s" % n.dtype)

def number_digits(n):
    return np.floor(np.log10(np.abs(n))) + 1

def _read_header(content):
    """Read HB-format header.

    Parameters
    ----------
    content: file-like
    """

    # First line
    line = content.readline()
    if not len(line) > 72:
        raise ValueError("Expected at least 72 characters for first line, "
                         "got: \n%s" % line)
    title = line[:72]
    key = line[72:]

    # Second line

    # totcrd|ptrcrd|indcrd|valcrd|rhscrd
    # totcrd: I14
    #   total number of lines
    # ptrcrd: I14
    #   number of lines for pointers
    # indcrd: I14
    #   number of lines for row or variables indices
    # valcrd: I14
    #   number of lines for values
    # rhscrd: I14
    #   number of lines for right hand side, starting guess and solutions (0 by
    #   default)
    line = content.readline()
    if not len(line.rstrip()) >= 56:
        raise ValueError("Expected at least 56 characters for second line, "
                         "got: \n%s" % line)
    tot_nlines = _expect_int(line[:14])
    ptr_nlines = _expect_int(line[14:28])
    ind_nlines = _expect_int(line[28:42])
    val_nlines = _expect_int(line[42:56])

    rhs_nlines = line[56:72].strip()
    if rhs_nlines == '':
        rhs_nlines = 0
    else:
        rhs_nlines = _expect_int(rhs_nlines)

    # Third line
    line = content.readline()
    if not len(line) >= 70:
        raise ValueError("Expected at least 72 character for third line, got:\n"
                         "%s" % line)

    mxtype = line[:3].upper()
    if not line[3:14] == " " * 11:
        raise ValueError("Malformed data for third line: %s" % line)
    if not len(mxtype) == 3:
        raise ValueError("mxtype expected to be 3 characters long")
    if not mxtype[0] == "R":
        raise ValueError("type %s not supported" % mxtype[0])

    n_rows = _expect_int(line[14:28])
    n_cols = _expect_int(line[28:42])
    n_zeros = _expect_int(line[42:56])
    n_elementals = _expect_int(line[56:70])
    if not n_elementals == 0:
        raise ValueError("Unexpected value %d for nltvl (last entry of line 3)"
                         % n_elementals)

    # Fourth line
    line = content.readline()

    parser = FortranFormatParser()
    ct = line.split()
    if not len(ct) == 3:
        raise ValueError("Expected 3 formats, got %s" % ct)

    ptr_fmt = parser.parse(ct[0])
    if not isinstance(ptr_fmt, IntFormat):
        raise ValueError("Expected int format for pointer format, got %s"
                         % ct[0])

    ind_fmt = parser.parse(ct[1])
    if not isinstance(ind_fmt, IntFormat):
        raise ValueError("Expected int format for indices format, got %s" %
                         ct[1])

    val_fmt = parser.parse(ct[2])
    if not isinstance(val_fmt, ExpFormat):
        raise ValueError("Expected exponential format for values, got %s"
                         % ct[2])
    header_info = HarwellBoeingHeader( title,
                        ptr_nlines, ptr_fmt.repeat, ptr_fmt.width,
                        ind_nlines, ind_fmt.repeat, ind_fmt.width,
                        val_nlines, val_fmt.repeat, val_fmt.width,
                        n_rows, n_cols, n_zeros,
                        key)
    return header_info

def read_hb(file):
    """Read a file in Harwell-Boeing format.

    Parameters
    ----------
    file: str-like or file-like
        if a string-like object, file is the name of the file to read. If a
        file-like object, the data are read from it.

    Returns
    -------
    m: sparse-matrix
        read sparse matrix
    """
    if isinstance(file, basestring):
        fid = open(file)
    else:
        fid = file

    try:
        return _read_hb(fid)
    finally:
        if isinstance(file, basestring):
            fid.close()

def _read_hb(content):
    """Read HB-format file.

    Parameters
    ----------
    content: file-like

    Returns
    -------
    m: sparse-matrix
    """
    header = HarwellBoeingHeader.from_file(content)

    # Number of bytes per data section, ignoring the last line, which
    # potentially contains few items than every other line
    ptr_nbytes = (header.pointer_nitems * header.pointer_width + 1) \
                 * (header.pointer_nlines - 1)
    ind_nbytes = (header.indices_nitems * header.indices_width + 1) \
                 * (header.indices_nlines - 1)
    val_nbytes = (header.values_nitems * header.values_width + 1) \
                 * (header.values_nlines - 1)

    ptr_string =  "".join([content.read(ptr_nbytes),
                           content.readline()])
    ptr = np.fromstring(ptr_string,
            dtype=np.int, sep=' ')

    ind_string = "".join([content.read(ind_nbytes),
                       content.readline()])
    ind = np.fromstring(ind_string,
            dtype=np.int, sep=' ')

    val_string = "".join([content.read(val_nbytes),
                          content.readline()])
    val = np.fromstring(val_string,
            dtype=np.float, sep=' ')

    try:
        return csc_matrix((val, ind-1, ptr-1),
                          shape=(header.n_rows, header.n_cols))
    except ValueError, e:
        raise e

def write_hb(file, m, title=None, key=None, fmt=None):
    """Write HB-format file.

    Parameters
    ----------
    file: str-like or file-like
        if a string-like object, file is the name of the file to read. If a
        file-like object, the data are read from it.
    m: sparse-matrix
        the sparse matrix to write
    title: str
        title put in the header
    key: str
        Key put in the header
    """
    # TODO: fix and document the format argument
    # fmt: dict
    #     dict containing the format for each pointer, indices and values
    #     arrays.
    if isinstance(file, basestring):
        fid = open(file, "w")
    else:
        fid = file

    try:
        _write_hb(fid, m, title, key, mxtype="RUA", fmt=fmt)
    finally:
        if isinstance(file, basestring):
            fid.close()

def _parse_fmt_argument(m, fmt, ptr, ind, val):
    # Compute formats
    if fmt is None:
        fmt = {}
    for k in ["ptr", "ind", "val"]:
        if not fmt.has_key(k):
            fmt[k] = "default"

    if fmt["ptr"] == "default":
        ptr_max = np.max(ptr)
        ptr_fmt = fortran_format(ptr_max)
    else:
        raise ValueError("Pointer format %s not supported" % fmt["ptr"])

    if fmt["ind"] == "default":
        ind_max = np.max(ind)
        ind_fmt = fortran_format(ind_max)
    else:
        raise ValueError("Index format %s not supported" % fmt["ind"])

    if fmt["val"] == "default":
        if m.dtype in [np.float32, np.float64]:
            prec = np.finfo(m.dtype).precision
        else:
            raise ValueError("dtype %s not supported for values" % m.dtype)
    else:
        prec = fmt["val"]["prec"]
    val_max = np.max(np.abs(val))
    val_fmt = fortran_format(val_max, prec=prec)

    return ptr_fmt, ind_fmt, val_fmt

def _write_hb(fid, m, title=None, key=None, mxtype="RUA", fmt=None):
    if title is None:
        title = "No Title"
    if len(title) > 72:
        raise ValueError("title cannot be > 72 characters")

    if key is None:
        key = "|No Key"
    if len(key) > 8:
        raise ValueError("key cannot be > 8 characters")

    header = [title.ljust(72) + key.ljust(8)]
    assert len(header[0]) == 80

    c = csc_matrix(m)
    val = c.data

    # At this point, we use fortran convention (one-indexing)
    ind = c.indices + 1
    ptr = c.indptr + 1

    ptr_fmt, ind_fmt, val_fmt = _parse_fmt_argument(m, fmt, ptr, ind, val)

    parser = FortranFormatParser()
    def _nrepeat(fmt):
        tp = parser.parse(fmt)
        return tp.repeat

    # *_n: number of items per (full) line
    ptr_n = _nrepeat(ptr_fmt)
    ind_n = _nrepeat(ind_fmt)
    val_n = _nrepeat(val_fmt)

    def _nlines(size, n):
        nlines = size / n
        if nlines * n != size:
            nlines += 1
        return nlines

    ptr_nlines = _nlines(ptr.size, ptr_n)
    ind_nlines = _nlines(ind.size, ind_n)
    val_nlines = _nlines(val.size, val_n)

    tot_nlines = ptr_nlines + ind_nlines + val_nlines

    mxtype = mxtype.upper()
    if np.iscomplexobj(val) and not mxtype[0] == "C":
        raise ValueError("Complex matrix, but given type is %s" % mxtype[0])
    elif not mxtype[0] in ["C", "R", "P"]:
        raise ValueError("value type %s not understood" % mxtype[0])

    if not mxtype[1] in ["U", "S", "H", "Z", "R"]:
        raise ValueError("Matrix type %s not understood" % mxtype[1])
    if not mxtype[2] in ["A", "E"]:
        raise ValueError("Matrix format %s not understood" % mxtype[2])

    header.append("%14d%14d%14d%14d" % 
                  (tot_nlines, ptr_nlines, ind_nlines, val_nlines))
    header.append("%14s%14d%14d%14d%14d" % 
                  (mxtype.ljust(14), m.shape[0],
                   m.shape[1], val.size, 0))
    header.append("%16s%16s%20s" %
                  (ptr_fmt.ljust(16), ind_fmt.ljust(16),
                   val_fmt.ljust(20)))

    def write_array(f, ar, nlines, n, ffmt):
        # ar_nlines is the number of full lines, n is the number of items per
        # line, ffmt the fortran format
        pyfmt = parser.to_python(ffmt)
        pyfmt_full = pyfmt * n

        # for each array to write, we first write the full lines, and special
        # case for partial line
        full = ar[:(nlines - 1) * n]
        for row in full.reshape((nlines-1, n)):
            f.write(pyfmt_full % tuple(row) + "\n")
        nremain = ar.size - full.size
        if nremain > 0:
            f.write((pyfmt * nremain) % tuple(ar[ar.size - nremain:]) + "\n")

    fid.write("\n".join(header))
    fid.write("\n")
    write_array(fid, ptr, ptr_nlines, ptr_n, ptr_fmt)
    write_array(fid, ind, ind_nlines, ind_n, ind_fmt)
    write_array(fid, val, val_nlines, val_n, val_fmt)
