"""
This script connects to the 3GPP FTP server to download specified standard files and stores them in the designated 'standards' folder.

"""

from ftplib import FTP, error_perm
import json
import os
import zipfile

class FTPClient:
    def __init__(self, host, user='', passwd=''):
        self.host = host
        self.ftp = FTP(host)  # Establish connection upon initialization
        try:
            self.ftp.login(user=user, passwd=passwd)
            print(f"Logged in to {self.host}")
        except Exception as e:
            print(f"Error connecting to {self.host}: {e}")
            raise

    def change_directory(self, path):
        """Change the working directory on the FTP server."""
        try:
            self.ftp.cwd(path)
            print(f"Changed to directory: {self.ftp.pwd()}")
        except error_perm as e:
            print(f"Permission error changing directory: {e}")
            raise
        except Exception as e:
            print(f"Error changing directory: {e}")
            raise
    
    def list_directory(self):
        dir_content = []
        try:
            self.ftp.retrlines('LIST', lambda line: dir_content.append(line))
        except Exception as e:
            print(f"Error listing directory: {e}")
            raise
        return dir_content

    def download(self, filename, local_path):

        # Ensure that `local_path` is a directory, and construct the full path to the file
        full_path = os.path.join(local_path, filename)
        with open (full_path, 'wb') as file:
            try:
                self.ftp.retrbinary(f'RETR {filename}', file.write)
                print(f"Downloaded {filename} to {local_path}")
            except Exception as e:
                print(f"An Error occured while downloading {filename} to {local_path}")
                #print(e)

    def close_connection(self):
        if self.ftp:
            self.ftp.close()
            print(f"Disconnected from {self.host}")


# std_list is path to the JSON file that contains the name of the files alongside their version
# local_path is path to the destination folder you want to download standards to
def get_standards(ftp_client: FTPClient, std_list: str, local_path: str):
    os.makedirs(local_path, exist_ok=True) # creating a directory if it does not exist.
    #open the json file
    with open(std_list, 'r') as series_list:
        series_data = json.load(series_list) # load the data from json file 
        series_found = False

        # check if a directory with given number exists 
        for entry in ftp_client.list_directory():
            if(series_data["series_no"] + "_series" in entry):
                ftp_client.change_directory(series_data["series_no"] + "_series")
                series_found = True
                break
        
        # Terminate the process if given series was not found
        if(not series_found):
            print(f'{series_data["series_no"]} was not found.')
            return
        
        # search for the given files in the series folder
        for entry in ftp_client.list_directory():
            for index in series_data["indexes"]: 
                if(series_data["series_no"] + "."+ index["name"] in entry):
                    ftp_client.change_directory(series_data["series_no"] + "." + index["name"]) # change directory to the current standard folder
                    # check for the version that needs to be downloaded
                    if(index['version'] == 'latest'):
                        all_versions = ftp_client.list_directory()
                        last_entry: str = all_versions[-1] # since in the 3Gpp files are organized by date latest is always the last entry of the directory
                        filename_ = last_entry.split()[-1] # Extract the name of the file from the entry
                        ftp_client.download(filename_, local_path)
                    else:
                        filename_ = series_data['series_no'] + index['name'] + '-' + index['version'] + ".zip"
                        ftp_client.download(filename_, local_path)
                    
                    ftp_client.change_directory("..") # going back one directory up after being finished with the current file 
        
        # getting back to the original path after donwloading all the standards 
        if(ftp_client.ftp.pwd() != directory_path):
            ftp_client.change_directory('..')
            

def unzip_all_in_folder(folder_path, extract_to): # This function is created by Chat-GPT
    os.makedirs(extract_to, exist_ok=True)
    # List all files in the folder
    for file_name in os.listdir(folder_path):
        # Construct full file path
        file_path = os.path.join(folder_path, file_name)
        
        # Check if it's a valid zip file
        if zipfile.is_zipfile(file_path):
            print(f'Unzipping {file_name}...')
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                # Extract the files to the destination folder
                zip_ref.extractall(extract_to)

            # removing the zipped file after it is unzipped to the 'extract_to' path 
            os.remove(file_path)
        else:
            print(f'Skipping {file_name}, not a zip file.')


#################### script #########################

try:
    host = "www.3gpp.org"
    directory_path = "Specs/archive" 
    download_folder = 'downloaded_standards'
    unzipped_folder = 'unzipped_standards'
    standards = ['23_series.json']
    # create the connection to FTP server and change to the wanted directory
    ftp_client = FTPClient(host)
    ftp_client.change_directory(directory_path)

    for standard in standards:
        get_standards(ftp_client, standard, download_folder)

    # unzip the downloaded standards 
    unzip_all_in_folder(download_folder, unzipped_folder)
finally:
    ftp_client.close_connection()



