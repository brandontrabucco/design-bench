import requests
import zipfile
import os


# the public url to objects available for download
SERVER_URL = "https://storage.googleapis.com/design-bench"


# the global path to a folder that stores all data files
DATA_DIR = os.path.join(
    os.path.abspath(
    os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), 'design_bench_data')


def get_confirm_token(response):
    """Get a confirmation token from the cookies associated with the
    google drive file download response

    """

    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value


def save_response(response, destination):
    """Save the response from google drive at a physical location in the disk
    assuming the destination is in a folder that exists

    """

    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)


def google_drive_download(download_target, disk_target):
    """Downloads a file from google drive using GET and stores that file
    at a specified location on the local disk

    Arguments:

    download_target: str
        the file id specified by google which is the 'X' in the url:
        https://drive.google.com/file/d/X/view?usp=sharing
    disk_target: str
        the destination for the file on this device, do not call this
        function is the file is already downloaded, as it will be overwritten

    Returns:

    success: bool
        a boolean that indicates whether the download was successful is True
        or an error was encountered when False (such as a 404 error)

    """

    # connect to google drive and request the file
    session = requests.Session()
    response = session.get("https://docs.google.com/uc?export=download",
                           params={'id': download_target}, stream=True)
    valid_response = response.status_code < 400
    if not valid_response:
        return valid_response

    # confirm that the download should start
    token = get_confirm_token(response)
    if token is not None:
        response = session.get("https://docs.google.com/uc?export=download",
                               params={'id': download_target,
                                       'confirm': token}, stream=True)
        valid_response = response.status_code < 400
        if not valid_response:
            return valid_response

    # save the content of the file to a local destination
    save_response(response, disk_target)
    return True


def direct_download(download_target, disk_target):
    """Downloads a file from a direct url using GET and stores that file
    at a specified location on the local disk

    Arguments:

    download_target: str
        the direct url where the file is located on a remote server
        available for direct download using GET
    disk_target: str
        the destination for the file on this device, do not call this
        function is the file is already downloaded, as it will be overwritten

    Returns:

    success: bool
        a boolean that indicates whether the download was successful is True
        or an error was encountered when False (such as a 404 error)

    """

    response = requests.get(download_target, allow_redirects=True)
    valid_response = response.status_code < 400
    if valid_response:
        with open(disk_target, "wb") as file:
            file.write(response.content)
    return valid_response


class DiskResource(object):
    """A resource manager that downloads files from remote destinations
    and loads these files from the disk, used to manage remote datasets
    for offline model-based optimization problems

    Public Attributes:

    is_downloaded: bool
        a boolean indicator that specifies whether this resource file
        is present at the specified location

    disk_target: str
        a string that specifies the location on disk where the target file
        is going to be placed

    download_target: str
        a string that gives the url or the google drive file id which the
        file is going to be downloaded from

    download_method: str
        the method of downloading the target file, which supports
        "google_drive" or "direct" for direct downloads

    Public Methods:

    get_data_path():
        Get a path to the file provided as an argument if it were inside the
        local folder used for storing downloaded resource files

    download():
        Download the remote file from either google drive or a direct
        remote url and store that file at a certain disk location

    """

    @staticmethod
    def get_data_path(file_path):
        """Get a path to the file provided as an argument if it were inside
        the local folder used for storing downloaded resource files

        Arguments:

        file_path: str
            a string that specifies the location relative to the data folder
            on disk where the target file is going to be placed

        Returns:

        data_path: str
            a string that specifies the absolute location on disk where the
            target file is going to be placed

        """

        return os.path.join(DATA_DIR, file_path)

    def __init__(self, disk_target, is_absolute=True,
                 download_target=None, download_method=None):
        """A resource manager that downloads files from remote destinations
        and loads these files from the disk, used to manage remote datasets
        for offline model-based optimization problems

        Arguments:

        disk_target: str
            a string that specifies the location on disk where the target file
            is going to be placed
        is_absolute: bool
            a boolean that indicates whether the provided disk_target path is
            an absolute path or relative to the data folder
        download_target: str
            a string that gives the url or the google drive file id which the
            file is going to be downloaded from
        download_method: str
            the method of downloading the target file, which supports
            "google_drive" or "direct" for direct downloads

        """

        self.disk_target = os.path.abspath(disk_target) \
            if is_absolute else DiskResource.get_data_path(disk_target)
        self.download_target = download_target
        self.download_method = download_method
        os.makedirs(os.path.dirname(self.disk_target), exist_ok=True)

    @property
    def is_downloaded(self):
        """a boolean indicator that specifies whether this resource file
        is present at the specified location

        """
        return os.path.exists(self.disk_target)

    def download(self, unzip=True):
        """Download the remote file from either google drive or a direct
        remote url and store that file at a certain disk location

        Arguments:

        unzip: bool
            a boolean indicator that specifies whether the downloaded file
            should be unzipped if the file extension is .zip

        Returns:

        success: bool
            a boolean that indicates whether the download was successful is True
            or an error was encountered when False (such as a 404 error)

        """

        # check that a download method for this file exists
        if (self.download_target is None
                or self.download_method is None):
            return False

        success = False

        # download using a direct method
        if self.download_method == "direct":
            success = direct_download(
                self.download_target, self.disk_target)

        # download using the google drive api
        elif self.download_method == "google_drive":
            success = google_drive_download(
                self.download_target, self.disk_target)

        # unzip the file if it is zipped
        if success and unzip and self.disk_target.endswith('.zip'):
            with zipfile.ZipFile(self.disk_target, 'r') as zip_ref:
                zip_ref.extractall(os.path.dirname(self.disk_target))

        return success
