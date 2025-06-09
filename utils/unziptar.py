import tarfile
import os

def unzip_all_targz(pth):
    """
    Extract all .tar.gz files in the given directory (non-recursively).
    Each archive is extracted into a subfolder with the same name (minus .tar.gz).
    """
    for filename in os.listdir(pth):
        if filename.endswith(".tar.gz"):
            tar_path = os.path.join(pth, filename)
            extract_folder = os.path.join(pth, filename[:-7])  # Remove ".tar.gz"
            print(f"Extracting: {tar_path} -> {extract_folder}")
            os.makedirs(extract_folder, exist_ok=True)
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(path=extract_folder)

# Example usage
pth = '/home/hux/datasets/h2odataset/dataset'  # Replace with your target directory
unzip_all_targz(pth)
