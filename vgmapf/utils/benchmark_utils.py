import timeit
import logging
import datetime
import functools

class time_it(object):
    def __init__(self, description = '', result_writer = None, log_level = logging.DEBUG ):
        self.log_level = log_level
        if result_writer is None and log_level is not None:
            def _result_writer(text):
                logging.getLogger(__name__).log(log_level, text)

            result_writer = _result_writer

        self.result_writer = result_writer
        self.description = description

        self.start_time = 0
        self.end_time = 0

    def __enter__(self):
        if self.result_writer:
            self.result_writer('STARTED %s\n' % self.description)
        self.start_time = timeit.default_timer()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = timeit.default_timer()
        elapsed_td = datetime.timedelta(seconds=self.end_time - self.start_time)
        if self.result_writer:
            self.result_writer('FINISHED %s ==> [!!!] Elapsed: %s [%dms]\n' % (
                self.description, elapsed_td, int(1000 * elapsed_td.seconds + elapsed_td.microseconds / 1000)))

    def getElapsed(self):
        return self.end_time - self.start_time

def time_this_func( description = None, result_writer = None, log_level = logging.DEBUG ):
    def _decorator( func ):
        @functools.wraps(func)
        def new_func( *args, **kwargs ):
            desc = description
            if desc is None:
                desc = 'Running %s' % func.__name__
            with time_it( desc, result_writer, log_level ):
                rc = func( *args, **kwargs )
            return rc

        return new_func
    return _decorator
