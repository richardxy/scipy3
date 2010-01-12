import ply.lex
import ply.yacc

#====================
#       Lexer
#====================
tokens = ("LPAR", "RPAR", "INT_ID", "INT", "EXP_ID", "DOT")

t_LPAR = r'\('
t_RPAR = r'\)'
t_INT_ID = r'I'
t_EXP_ID = r'E'
t_DOT = r'\.'

def t_INT(t):
    r'\d+'
    t.value = int(t.value)
    return t

def t_error(t):
    print "Error while scanning at line %d, col %d -> %s" % \
          (t.lineno, t.lexpos, t.value)
    t.lexer.skip(1)

#====================
#       Grammar
#====================
def p_format(p):
    "format : LPAR format_string RPAR"
    p[0] = p[2]

def p_format_string(p):
    """\
    format_string : repeated_format
                  | single_format
    """
    p[0] = p[1]

def p_repeated_format(p):
    """\
    repeated_format : INT single_format
    """
    p[0] = Repeated(p[1], p[2])

def p_single_format(p):
    """\
    single_format : int_format
                  | exp_format
    """
    p[0] = p[1]

def p_int_format(p):
    "int_format : INT_ID INT"
    p[0] = Int(p[2])

def p_exp_format(p):
    """\
    exp_format : simple_exp_format
               | exp_exp_format
    """
    p[0] = p[1]

def p_simple_exp_format(p):
    "simple_exp_format : EXP_ID INT DOT INT"
    p[0] = Exp(significand=p[4], width=p[2])

def p_exp_exp_format(p):
    "exp_exp_format : EXP_ID INT DOT INT EXP_ID INT"
    # Exp format with specified number of digits in exponent
    p[0] = Exp(significand=p[4], width=p[2], exponent=p[6])

def p_error(p):
    msg = "Error while parsing token '%s' at line %d, column %d" % \
          (p.value, p.lineno, p.lexpos)
    raise SyntaxError(msg)

#===========
#   AST
#===========
class Node(object):
    pass

class Int(Node):
    def __init__(self, width):
        self.type = "INT"
        self.width = width

    def __str__(self):
        return "INT (width=%d)" % self.width

class Exp(Node):
    def __init__(self, width, significand, exponent=None):
        self.type = "EXP"
        self.significand = significand
        self.width = width
        self.exponent = exponent

    def __str__(self):
        msg = "EXP with %d significand digits (width=%d" \
              % (self.width, self.significand)
        if self.exponent is not None:
            msg += ", >=%d digits in exponent" % self.exponent
        msg += ")"
        return msg

class Repeated(Node):
    def __init__(self, repeated, value):
        self.repeated = repeated
        self.value = value

    def __str__(self):
        return "%d repeat of '%s'" % (self.repeated, self.value)

#====================
#   Public API
#====================
# Errors
class BadFortranFormat(Exception):
    pass

class Format(object):
    pass

class IntFormat(object):
    def __init__(self, width, repeat=1):
        self.width = width
        self.repeat= repeat

class ExpFormat(object):
    def __init__(self, width, significand, exponent=None, repeat=1):
        self.width = width
        self.significand = significand
        self.repeat = repeat
        self.exponent = exponent

class FortranFormatParser(object):
    def __init__(self):
        self.lexer = ply.lex.lex()
        self.parser = ply.yacc.yacc(start="format")

    def parse(self, s):
        try:
            res = self.parser.parse(s)
        except SyntaxError, e:
            raise BadFortranFormat(str(e))
        if isinstance(res, Repeated):
            repeat = res.repeated
            res = res.value
        else:
            repeat = 1
        if isinstance(res, Int):
            return IntFormat(res.width, repeat)
        elif isinstance(res, Exp):
            return ExpFormat(res.width, res.significand, res.exponent, repeat)

    def to_python(self, fmt):
        fmt = fmt.upper().strip()
        tp = self.parse(fmt)
        if isinstance(tp, IntFormat):
            return "%" + str(tp.width) + "d"
        elif isinstance(tp, ExpFormat):
            return "%" + str(tp.width) + "." + str(tp.significand) + "e"
        else:
            raise ValueError("Unsupported format: %s" % fmt)
