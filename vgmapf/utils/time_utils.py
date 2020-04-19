import time
import datetime

DEFAULT_STRFTIME_FORMAT = '%Y-%m-%d_%H-%M-%S'
DATE_STRFTIME_FORMAT = '%Y-%m-%d'

def get_time_stamp( time_t, gmt_timestamp = False, with_micro_seconds = False, strftime_format = DEFAULT_STRFTIME_FORMAT ):
#    tm = time.localtime( time_t )
    if gmt_timestamp:
        dt = datetime.datetime.utcfromtimestamp( time_t )
    else:
        dt = datetime.datetime.fromtimestamp( time_t )

    if with_micro_seconds:
        strftime_format += '.%f'
    return dt.strftime( strftime_format )


def get_current_time_stamp( gmt_timestamp = False, with_micro_seconds = False ):
    return get_time_stamp( time.time(), gmt_timestamp, with_micro_seconds  )
