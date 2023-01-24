from datetime import timedelta
import dateutil.parser
import pandas as pd
import numpy as np


def hour_of_week(time_stamp, timezone_offset=None):
    """
    Generate an hour of week value from a timestamp and timesone offset
    
    :param str time_stamp: a timestamp
    :param str timezone_offset: a timezone offset
    :return: a numeric string representing the hour of week
    :rtype: str
    """
    timestamp = dateutil.parser.parse(time_stamp)
    if timezone_offset is not None:
        hour_offset = int(timezone_offset.replace('\\', '').replace('N', '0'))
        timestamp += timedelta(hours=hour_offset)

    week_day = (timestamp.weekday() + 1)  % 7 # offset to make Sunday the first day of week - WHY?!
    hour = timestamp.hour
    return str(week_day * 24 + hour)


def hour_of_week_from_df(time_stamp, timezone_offset):
    """
    Generate an hour of week series from a timestamp and timesone offset series
    
    :param pandas.Series time_stamp: a timestamp
    :param pandas.Series timezone_offset: a timezone offset
    :return: a numeric string representing the hour of week as series
    :rtype: pandas.Series
    """
    timeStamp = pd.to_datetime(time_stamp)
    timezoneOffset = timezone_offset.str.replace('\\', '').replace('N', '0').astype(np.int8).fillna(0)
    timeStamp += pd.to_timedelta(timezoneOffset, unit='h')
    week_day = (timeStamp.dt.dayofweek + 1) % 7
    hour = timeStamp.dt.hour
    return (week_day * 24 + hour).astype(str)
