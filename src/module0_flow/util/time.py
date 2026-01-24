import re
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np

def find_closest_timestamp(array_tstamp, charge_name):
    print(charge_name)

    match = re.search(r"(\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})", charge_name)
    if not match:
        raise ValueError(f"No timestamp found in filename: {charge_name} cannot extract elifetime")
    
    ts_str = match.group(1)
    if 'CET' in charge_name:
        tz = ZoneInfo("Europe/Paris")
    elif 'CDT' in charge_name:
        tz = ZoneInfo("America/Chicago")
    elif 'CST' in charge_name:
        tz = ZoneInfo("America/Chicago")
    else:
        tz = ZoneInfo("UTC")
        
    file_dt = datetime.strptime(ts_str, "%Y_%m_%d_%H_%M_%S").replace(tzinfo=tz).timestamp()

    array_tstamp = np.sort(array_tstamp)
    candidate_i = np.argmin(np.abs(array_tstamp - file_dt))

    return str(array_tstamp[candidate_i]) # Select the closest timestamp from the file timestamp
