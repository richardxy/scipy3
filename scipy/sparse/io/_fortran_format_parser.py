import re
import warnings

import numpy

__all__ = ["BadFortranFormat", "FortranFormatParser", "IntFormat", "ExpFormat"]

TOKENS = {
    "LPAR": r"\(",
    "RPAR": r"\)",
    "INT_ID": r"I",
    "EXP_ID": r"E",
    "INT": r"\d+",
    "DOT": r"\.",
}

class BadFortranFormat(SyntaxError):
    pass

class FormatOverflow(Warnings):
    pass

def number_digits(n):
    return np.floor(np.log10(np.abs(n))) + 1

def fortran_format_int(n, min=None):
    ndigits = number_digits(n)
    if min is None:
        min = ndigits
    nitems = 80 / (ndigits + 1)

class IntFormat(object):
    def __init__(self, width, min=None, repeat=None):
        self.width = width
        self.repeat = repeat
        self.min = min

    def __repr__(self):
        r = ""
        if self.repeat:
            r += str(self.repeat)
        r += "I%d" % self.width
        if self.min:
            r += ".%d" % self.min
        return r

class ExpFormat(object):
    def __init__(self, width, significand, min=None, repeat=None):
        self.width = width
        self.significand = significand
        self.repeat = repeat
        self.min = min

    def __repr__(self):
        r = ""
        if self.repeat:
            r += str(self.repeat)
        r += "E%d.%d" % (self.width, self.significand)
        if self.min:
            r += "E%d" % self.min
        return r

class Token(object):
    def __init__(self, type, value, pos):
        self.type = type
        self.value = value
        self.pos = pos

    def __str__(self):
        return """Token('%s', "%s")""" % (self.type, self.value)

    def __repr__(self):
        return self.__str__()

class Tokenizer(object):
    def __init__(self):
        self.tokens = TOKENS.keys()
        self.res = [re.compile(TOKENS[i]) for i in self.tokens]

    def input(self, s):
        self.data = s
        self.curpos = 0
        self.len = len(s)

    def next_token(self):
        curpos = self.curpos
        tokens = self.tokens

        while curpos < self.len:
            for i, r in enumerate(self.res):
                m = r.match(self.data, curpos)
                if m is None:
                    continue
                else:
                    self.curpos = m.end()
                    return Token(self.tokens[i], m.group(), self.curpos)
            else:
                raise SyntaxError("Unknown character at position %d (%s)" \
                                  % (self.curpos, self.data[curpos]))

# Grammar for fortran format:
# format            : LPAR format_string RPAR
# format_string     : repeated | simple
# repeated          : repeat simple
# simple            : int_fmt | exp_fmt
# int_fmt           : INT_ID width
# exp_fmt           : simple_exp_fmt
# simple_exp_fmt    : EXP_ID width DOT significand
# extended_exp_fmt  : EXP_ID width DOT significand EXP_ID ndigits
# repeat            : INT
# width             : INT
# significand       : INT
# ndigits           : INT

# Naive fortran formatter - parser is hand-made
class FortranFormatParser(object):
    def __init__(self):
        self.tokenizer = Tokenizer()

    def parse(self, s):
        self.tokenizer.input(s)

        tokens = []

        try:
            while True:
                t = self.tokenizer.next_token()
                if t is None:
                    break
                else:
                    tokens.append(t)
            return self._parse_format(tokens)
        except SyntaxError, e:
            raise BadFortranFormat(str(e))

    def _get_min(self, tokens):
        next = tokens.pop(0)
        if not next.type == "DOT":
            raise SyntaxError()
        next = tokens.pop(0)
        return next.value

    def _expect(self, token, tp):
        if not token.type == tp:
            raise SyntaxError()

    def _parse_format(self, tokens):
        if not tokens[0].type == "LPAR":
            raise SyntaxError("Expected left parenthesis at position "\
                              "%d (got '%s')" % (0, tokens[0].value))
        elif not tokens[-1].type == "RPAR":
            raise SyntaxError("Expected right parenthesis at position "\
                              "%d (got '%s')" % (len(tokens), tokens[-1].value))

        tokens = tokens[1:-1]
        types = [t.type for t in tokens]
        if types[0] == "INT":
            repeat = int(tokens.pop(0).value)
        else:
            repeat = None

        next = tokens.pop(0)
        if next.type == "INT_ID":
            next = self._next(tokens, "INT")
            width = int(next.value)
            if tokens:
                min = int(self._get_min(tokens))
            else:
                min = None
            return IntFormat(width, min, repeat)
        elif next.type == "EXP_ID":
            next = self._next(tokens, "INT")
            width = int(next.value)

            next = self._next(tokens, "DOT")

            next = self._next(tokens, "INT")
            significand = int(next.value)

            if tokens:
                next = self._next(tokens, "EXP_ID")

                next = self._next(tokens, "INT")
                min = int(next.value)
            else:
                min = None
            return ExpFormat(width, significand, min, repeat)
        else:
            raise SyntaxError("Invalid formater type %s" % next.value)

    def _next(self, tokens, tp):
        if not len(tokens) > 0:
            raise SyntaxError()
        next = tokens.pop(0)
        self._expect(next, tp)
        return next

    def to_python(self, fmt):
        fmt = fmt.upper().strip()
        tp = self.parse(fmt)
        if isinstance(tp, IntFormat):
            return "%" + str(tp.width) + "d"
        elif isinstance(tp, ExpFormat):
            return "%" + str(tp.width) + "." + str(tp.significand) + "e"
        else:
            raise ValueError("Unsupported format: %s" % fmt)
