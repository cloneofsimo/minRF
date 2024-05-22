import os
import time
from huggingface_hub import HfApi, CommitOperationAdd, create_branch

if False:
    CKPT_DIRS = "./data"
    SLEEP_INTERVAL = 10
    REPO_ID = "cloneofsimo/test_model"
    REPO_TYPE = "model"
else:
    CKPT_DIRS = "/home/host/simo/ckpts/5b_highres"
    SLEEP_INTERVAL = 60
    REPO_ID = "cloneofsimo/lavenderflow-5.6B"
    REPO_TYPE = "model"

# Initialize the API
api = HfApi()


def get_folder_size(folder_path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):
                total_size += os.path.getsize(fp)
    return total_size


def upload_if_stable(folder_path, relpath, wait_time=300):
    """Waits for the folder size to stabilize before uploading."""
    size1 = get_folder_size(folder_path)
    time.sleep(wait_time)
    size2 = get_folder_size(folder_path)

    bname = f"highres-{relpath}"

    if size1 == size2:
        print(f"Uploading {folder_path} to Hugging Face Hub.")
        try:
            create_branch(REPO_ID, repo_type=REPO_TYPE, branch=bname)
        except:
            pass

        api.upload_folder(
            folder_path=folder_path,
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            revision=bname,
        )
        print(f"Uploaded {folder_path} successfully.")

        # delete the folder
        os.system(f"rm -rf {folder_path}")
        return True

    return False


def monitor_ckpt_dirs():
    known_folders = set()

    while True:
        current_folders = set(os.listdir(CKPT_DIRS))
        new_folders = current_folders - known_folders

        new_folders = list(new_folders)
        # sort based on model_xxxx
        new_folders.sort(key=lambda x: int(x.split("_")[1]), reverse=True)

        for folder in new_folders:
            folder_path = os.path.join(CKPT_DIRS, folder)
            if os.path.isdir(folder_path):
                print(f"Detected new folder: {folder}")
                relpath = os.path.relpath(folder_path, CKPT_DIRS)
                if upload_if_stable(folder_path, relpath, SLEEP_INTERVAL):
                    known_folders.add(folder)

        time.sleep(SLEEP_INTERVAL)


if __name__ == "__main__":
    print("Starting to monitor for new model directories.")
    monitor_ckpt_dirs()
