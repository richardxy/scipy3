"""
Implementation of Harwell-Boeing read/write.
"""
# Really crude at the moment:
#   - only support for assembled, non-symmetric, real matrices
#   - only support integer for pointer/indices
#   - only support exponential format for values

# TODO:
#   - support for more value types (other float format at least)
#   - API for format specifier in hb_write ?
#   - Add more support (symmetric/complex matrices, non-assembled matrices ?)

# XXX: reading is reasonably efficient (>= 85 % is in numpy.fromstring), but
# takes a lot of memory. Being faster would require compiled code.
# write_hb is not efficient. Although not a terribly exciting task,
# having reusable facilities to efficiently read/write fortran-formatted files
# would be useful outside this module.
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

def _nbytes_full(fmt, nlines):
    """Return the number of bytes to read to get every full lines for the
    given parsed fortran format."""
    return (fmt.repeat * fmt.width + 1) * (nlines - 1)

class HBHeader(object):
    @classmethod
    def from_data(cls, title, key, pointer, indices, values, fmt=None):
        if fmt is None:
            pointer_max = np.max(pointer)
        else:
            raise NotImplementedError("fmt argument not supported yet.")

    @classmethod
    def from_file(cls, fid):
        # First line
        line = fid.readline()
        if not len(line) > 72:
            raise ValueError("Expected at least 72 characters for first line, "
                             "got: \n%s" % line)
        title = line[:72]
        key = line[72:]

        # Second line
        line = fid.readline()
        if not len(line.rstrip()) >= 56:
            raise ValueError("Expected at least 56 characters for second line, "
                             "got: \n%s" % line)
        total_nlines    = _expect_int(line[:14])
        pointer_nlines  = _expect_int(line[14:28])
        indices_nlines  = _expect_int(line[28:42])
        values_nlines   = _expect_int(line[42:56])

        rhs_nlines = line[56:72].strip()
        if rhs_nlines == '':
            rhs_nlines = 0
        else:
            rhs_nlines = _expect_int(rhs_nlines)
        if not rhs_nlines == 0:
            raise ValueError("Only files without right hand side supported for " \
                             "now.")

        # Third line
        line = fid.readline()
        if not len(line) >= 70:
            raise ValueError("Expected at least 72 character for third line, got:\n"
                             "%s" % line)

        mxtype_s = line[:3].upper()
        if not len(mxtype_s) == 3:
            raise ValueError("mxtype expected to be 3 characters long")

        mxtype = HBMatrixType.from_fortran(mxtype_s)
        if not mxtype.value_type in ["real", "integer"]:
            raise ValueError("Only real or integer matrices supported for now")
        if not mxtype.structure == "unsymmetric":
            raise ValueError("Only unsymmetric matrices supported for now")
        if not mxtype.storage == "assembled":
            raise ValueError("Only assembled matrices supported for now")

        if not line[3:14] == " " * 11:
            raise ValueError("Malformed data for third line: %s" % line)

        nrows = _expect_int(line[14:28])
        ncols = _expect_int(line[28:42])
        nnon_zeros = _expect_int(line[42:56])
        nelementals = _expect_int(line[56:70])
        if not nelementals == 0:
            raise ValueError("Unexpected value %d for nltvl (last entry of line 3)"
                             % nelementals)

        # Fourth line
        line = fid.readline()

        ct = line.split()
        if not len(ct) == 3:
            raise ValueError("Expected 3 formats, got %s" % ct)

        return cls(title, key,
                   total_nlines, pointer_nlines, indices_nlines, values_nlines,
                   mxtype, nrows, ncols, nnon_zeros,
                   ct[0], ct[1], ct[2],
                   rhs_nlines, nelementals)

    def __init__(self, title, key,
            total_nlines, pointer_nlines, indices_nlines, values_nlines,
            mxtype, nrows, ncols, nnon_zeros,
            pointer_format_str, indices_format_str, values_format_str,
            right_hand_sides_nlines=0, nelementals=0):
        """Do not use this directly, but the class ctrs (from_* functions)."""
        self.title = title
        self.key = key
        if title is None:
            title = "No Title"
        if len(title) > 72:
            raise ValueError("title cannot be > 72 characters")

        if key is None:
            key = "|No Key"
        if len(key) > 8:
            raise ValueError("key cannot be > 8 characters")


        self.total_nlines = total_nlines
        self.pointer_nlines = pointer_nlines
        self.indices_nlines = indices_nlines
        self.values_nlines = values_nlines

        parser = FortranFormatParser()
        pointer_format = parser.parse(pointer_format_str)
        if not isinstance(pointer_format, IntFormat):
            raise ValueError("Expected int format for pointer format, got %s"
                             % pointer_format)

        indices_format = parser.parse(indices_format_str)
        if not isinstance(indices_format, IntFormat):
            raise ValueError("Expected int format for indices format, got %s" %
                             indices_format)

        values_format = parser.parse(values_format_str)
        if isinstance(values_format, ExpFormat):
            if not mxtype.value_type in ["real", "complex"]:
                raise ValueError("Inconsistency between matrix type %s and " \
                                 "value type %s" % (mxtype, values_format))
            values_dtype = np.float64
        elif isinstance(val_fmt, IntFormat):
            if not mxtype.value_type in ["integer"]:
                raise ValueError("Inconsistency between matrix type %s and " \
                                 "value type %s" % (mxtype, values_format))
            # XXX: fortran int -> dtype association ?
            values_dtype = np.int32
        else:
            raise ValueError("Unsupported format for values %s" % ct[2])
        self.pointer_dtype = np.int32
        self.indices_dtype = np.int32
        self.values_dtype = values_dtype

        self.pointer_nlines = pointer_nlines
        self.pointer_nbytes_full = _nbytes_full(pointer_format, pointer_nlines)

        self.indices_nlines = indices_nlines
        self.indices_nbytes_full = _nbytes_full(indices_format, indices_nlines)

        self.values_nlines = values_nlines
        self.values_nbytes_full = _nbytes_full(values_format, values_nlines)

        self.nrows = nrows
        self.ncols = ncols
        self.nnon_zeros = nnon_zeros
        self.nelementals = nelementals

def _expect_int(value, msg=None):
    try:
        return int(value)
    except ValueError:
        if msg is None:
            msg = "Expected an int, got %s"
        raise ValueError(msg % value)

def _read_hb_data(content, header):
    # XXX: look at a way to reduce memory here (big string creation)
    ptr_string =  "".join([content.read(header.pointer_nbytes_full),
                           content.readline()])
    ptr = np.fromstring(ptr_string,
            dtype=np.int, sep=' ')

    ind_string = "".join([content.read(header.indices_nbytes_full),
                       content.readline()])
    ind = np.fromstring(ind_string,
            dtype=np.int, sep=' ')

    val_string = "".join([content.read(header.values_nbytes_full),
                          content.readline()])
    val = np.fromstring(val_string,
            dtype=header.values_dtype, sep=' ')

    try:
        return csc_matrix((val, ind-1, ptr-1),
                          shape=(header.nrows, header.ncols))
    except ValueError, e:
        raise e

class HBMatrixType(object):
    """Class to hold the matrix type."""
    # q2f* translates qualified names to fortran character
    _q2f_type = {
        "real": "R",
        "complex": "C",
        "pattern": "P",
        "integer": "I",
    }
    _q2f_structure = {
            "symmetric": "S",
            "unsymmetric": "U",
            "hermitian": "H",
            "skewsymmetric": "Z",
            "rectangular": "R"
    }
    _q2f_storage = {
        "assembled": "A",
        "elemental": "E",
    }

    _f2q_type = dict([(j, i) for i, j in _q2f_type.items()])
    _f2q_structure = dict([(j, i) for i, j in _q2f_structure.items()])
    _f2q_storage = dict([(j, i) for i, j in _q2f_storage.items()])

    @classmethod
    def from_fortran(cls, fmt):
        if not len(fmt) == 3:
            raise ValueError("Fortran format for matrix type should be 3 " \
                             "characters long")
        try:
            value_type = cls._f2q_type[fmt[0]]
            structure = cls._f2q_structure[fmt[1]]
            storage = cls._f2q_storage[fmt[2]]
            return cls(value_type, structure, storage)
        except KeyError:
            raise ValueError("Unrecognized format %s" % fmt)

    def __init__(self, value_type, structure, storage="assembled"):
        self.value_type = value_type
        self.structure = structure
        self.storage = storage

        if not value_type in self._q2f_type.keys():
            raise ValueError("Unrecognized type %s" % value_type)
        if not structure in self._q2f_structure.keys():
            raise ValueError("Unrecognized structure %s" % structure)
        if not storage in self._q2f_storage.keys():
            raise ValueError("Unrecognized storage %s" % storage)

    def fortran_fmt(self):
        return self._q2f_type[self.value_type] + \
               self._q2f_structure[self.structure] + \
               self._q2f_storage[self.storage]

    def __repr__(self):
        return "HBMatrixType(%s, %s, %s)" % \
               (self.value_type, self.structure, self.storage)

class HBFile(object):
    def __init__(self, file, hb_info=None):
        """Create a HBFile instance.

        Parameters
        ----------
        file: file-object
            StringIO work as well
        hb_info: HBHeader
            Should be given as an argument for writing, in which case the file
            should be writable.
        """
        self._fid = file
        if hb_info is None:
            self._hb_info = HBHeader.from_file(file)
        else:
            if not file.writable():
                raise IOError("file %s is not writable, and hb_info "
                              "was given." % file)

    @property
    def title(self):
        return self._hb_info.title

    @property
    def key(self):
        return self._hb_info.key

    @property
    def type(self):
        return self._hb_info.mxtype.value_type

    @property
    def structure(self):
        return self._hb_info.mxtype.structure

    @property
    def storage(self):
        return self._hb_info.mxtype.storage

    def read_matrix(self):
        return _read_hb_data(self._fid, self._hb_info)

def read_hb(file):
    """Read HB-format file.

    Parameters
    ----------
    file: str-like or file-like
        if a string-like object, file is the name of the file to read. If a
        file-like object, the data are read from it.
    """
    def _get_matrix(fid):
        hb = HBFile(fid)
        return hb.read_matrix()

    if isinstance(file, basestring):
        fid = open(file)
        try:
            return _get_matrix(fid)
        finally:
            fid.close()
    else:
        return _get_matrix(file)

def write_hb(file, m, hb_info):
    """Write HB-format file.

    Parameters
    ----------
    file: str-like or file-like
        if a string-like object, file is the name of the file to read. If a
        file-like object, the data are read from it.
    m: sparse-matrix
        the sparse matrix to write
    hb_info: HBInfo
        contains the meta-data for write_hb
    """
    def _set_matrix(fid):
        hb = HBFile(fid, hb_info)
        return hb.write_matrix(m)

    if isinstance(file, basestring):
        fid = open(file, "w")
        try:
            return _set_matrix(fid)
        finally:
            fid.close()
    else:
        return _set_matrix(file)
