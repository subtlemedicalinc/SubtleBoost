import os
import sys
import json
import datetime
from subtle.util import licensing

def get_license(app, serial_number, duration, fname_out):
    app_dict = {
        "SubtleHello": 1000,
        "SubtlePET": 2000,
        "SubtleMR": 3000,
        "SubtleGAD": 4000,
        "app-utilities": 9000
    }

    date_format = '%m/%d/%Y'
    exp_date = datetime.date.strftime(datetime.date.today() + duration, date_format)

    license_dict = licensing.generate_license(app_dict[app], app, serial_number, exp_date)

    with open(fname_out, 'w') as f:
        json.dump(license_dict, f)

if __name__ == "__main__":
    path_out = sys.argv[1]
    fname_out = os.path.join(path_out, "license_gad.json")

    serial_number="test"
    app = "SubtleGAD"
    duration = datetime.timedelta(days=2)
    get_license(app, serial_number, duration, fname_out)
