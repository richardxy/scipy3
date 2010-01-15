def number_digits(n):
    return np.floor(np.log10(np.abs(n))) + 1

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

def _write_hb(fid, m, hb_info):
    header = [hb_info.title.ljust(72) + hb_info.key.ljust(8)]

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

