from datetime import timedelta, datetime
import calendar

def calculate_interval_return(current_value, base_value):
    if base_value == 0:
        return None
    return (current_value - base_value) / base_value

def annualized_return(rt, delta_days):
    if delta_days > 0 and rt is not None:
        return rt * (365 / delta_days)
    return None

def get_quarter_days(current_date):
    prev_quarter = ((current_date.month - 1) // 3 + 1) - 1
    if prev_quarter == 0:
        prev_quarter = 4
        year = current_date.year - 1
    else:
        year = current_date.year

    quarter_to_month = {1: 3, 2: 6, 3: 9, 4: 12}
    last_month_of_quarter = quarter_to_month[prev_quarter]
    _, last_day_of_quarter = calendar.monthrange(year, last_month_of_quarter)
    return datetime(year, last_month_of_quarter, last_day_of_quarter)