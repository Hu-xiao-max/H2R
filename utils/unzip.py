import zipfile
import os

def unzip_all_zips(pth):
    """
    Unzips all .zip files under the given directory (non-recursively).
    Each zip file is extracted into a folder with the same name (without .zip).
    """
    for filename in os.listdir(pth):
        if filename.endswith(".zip"):
            zip_path = os.path.join(pth, filename)
            extract_folder = os.path.join(pth, filename[:-4])  # Remove ".zip"
            print(f"Unzipping: {zip_path} -> {extract_folder}")
            os.makedirs(extract_folder, exist_ok=True)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_folder)

# Example usage
pth = '/home/hux/datasets/h2odataset/dataset'  # Replace with your target directory
unzip_all_zips(pth)
