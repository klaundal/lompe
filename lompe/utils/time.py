""" time tools """
import numpy as np
import pandas as pd


def date_to_doy(month, day, leapyear = False):
    """ return day of year (DOY) at given month, day

        month and day -- can be arrays, but must have equal shape
        leapyear      -- can be array of equal shape or scalar

        return value  --  doy, with same shape as month and day
                          but always an array: shape (1,) if input is scalar

        The code is vectorized, so it should be relatively fast. 

        KML 2016-04-20
    """

    month = np.array(month, ndmin = 1)
    day   = np.array(day, ndmin = 1)

    if type(leapyear) == bool:
        leapyear = np.full_like(day, leapyear, dtype = bool)

    # check that shapes match
    if month.shape != day.shape:
        raise ValueError('date2ody: month and day must have the same shape')

    # check that month in [1, 12]
    if month.min() < 1 or month.max() > 12:
        raise ValueError('month not in [1, 12]')

    # check if day < 1
    if day.min() < 1:
        raise ValueError('date2doy: day must not be less than 1')

    # flatten arrays:
    shape = month.shape
    month = month.flatten()
    day   = day.flatten()

    # check if day exceeds days in months
    days_in_month    = np.array([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    days_in_month_ly = np.array([0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    if ( (np.any(day[~leapyear] > days_in_month   [month[~leapyear]])) | 
         (np.any(day[ leapyear] > days_in_month_ly[month[ leapyear]])) ):
        raise ValueError('date2doy: day must not exceed number of days in month')

    cumdaysmonth = np.cumsum(days_in_month[:-1])

    # day of year minus possibly leap day:
    doy = cumdaysmonth[month - 1] + day
    # add leap day where appropriate:
    doy[month >= 3] = doy[month >= 3] + leapyear[month >= 3]

    return doy.reshape(shape)


def is_leapyear(year):
    """ Check for leapyear (handles arrays and preserves shape)

    """

    # if array:
    if type(year) is np.ndarray:
        out = np.full_like(year, False, dtype = bool)

        out[ year % 4   == 0] = True
        out[ year % 100 == 0] = False
        out[ year % 400 == 0] = True

        return out

    # if scalar:
    if year % 400 == 0:
        return True

    if year % 100 == 0:
        return False

    if year % 4 == 0:
        return True

    else:
        return False


def yearfrac_to_datetime(fracyear):
    """ 
    Convert fraction of year to datetime 

    Parameters
    ----------
    fracyear : iterable
        Date(s) in decimal year. E.g., 2021-03-28 is 2021.2377
        Must be an array, list or similar.

    Returns
    -------
    datetimes : array
        Array of datetimes
    """

    year = np.uint16(fracyear) # truncate fracyear to get year
    # use pandas TimedeltaIndex to represent time since beginning of year: 
    delta_year = pd.to_timedelta((fracyear - year)*(365 + is_leapyear(year)), unit = 'D')
    # and DatetimeIndex to represent beginning of years:
    start_year = pd.DatetimeIndex(list(map(str, year)))
 
    # adding them produces the datetime:
    return (start_year + delta_year).to_pydatetime()
