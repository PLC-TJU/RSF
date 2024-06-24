# Authors: Divyesh Narayanan <divyesh.narayanan@gmail.com>
#
# License: BSD (3-clause)

import os.path as osp

from mne import get_config

from moabb.utils import set_download_dir

original_path = get_config("MNE_DATA")
print(f"The download directory is currently {original_path}")
new_path = osp.join(osp.expanduser("~"), "mne_data_test")
set_download_dir(new_path)

check_path = get_config("MNE_DATA")
print(f"Now the download directory has been changed to {check_path}")

set_download_dir(original_path)