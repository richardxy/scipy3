from numpy.testing \
    import \
        TestCase, assert_equal, assert_raises

from scipy.sparse.io._fortran_format_parser \
    import \
        FortranFormatParser, IntFormat, ExpFormat, BadFortranFormat

class TestFormat(TestCase):
    def setUp(self):
        self.parser = FortranFormatParser()

    def _test_equal(self, format, ref):
        ret = self.parser.parse(format)
        assert_equal(ret.__dict__, ref.__dict__)

    def test_simple_int(self):
        self._test_equal("(I4)", IntFormat(4))

    def test_simple_repeated_int(self):
        self._test_equal("(3I4)", IntFormat(4, repeat=3))

    def test_simple_exp(self):
        self._test_equal("(E4.3)", ExpFormat(4, 3))

    def test_exp_exp(self):
        self._test_equal("(E8.3E3)", ExpFormat(8, 3, 3))

    def test_repeat_exp(self):
        self._test_equal("(2E4.3)", ExpFormat(4, 3, repeat=2))

    def test_repeat_exp_exp(self):
        self._test_equal("(2E8.3E3)", ExpFormat(8, 3, 3, repeat=2))

    def test_wrong_formats(self):
        def _test_invalid(bad_format):
            assert_raises(BadFortranFormat, lambda: self.parser.parse(bad_format))
        _test_invalid("I4")
        _test_invalid("(E4)")
        _test_invalid("(E4.)")
        _test_invalid("(E4.E3)")
