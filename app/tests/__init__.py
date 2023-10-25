# This is required to bypass the @processify decorator while unit testing
from subtle.util import multiprocess_utils
multiprocess_utils.processify = (lambda func: func)
