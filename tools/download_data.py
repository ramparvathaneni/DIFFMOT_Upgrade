from datasets import load_dataset
import os, shutil

# Choose where to save the dataset
save_dir = "/home/tsrp/datasets/DanceTrack"   # <-- change to your path
os.makedirs(save_dir, exist_ok=True)

# Load dataset from Hugging Face
dataset = load_dataset("Voxel51/DanceTrack")

# The dataset contains three splits: train, val, and test
for split in ["train", "val", "test"]:
    print(f"Downloading {split} split...")
    ds_split = dataset[split]
    out_dir = os.path.join(save_dir, split)
    os.makedirs(out_dir, exist_ok=True)

    # Each entry has sequence metadata and image/annotation files
    for i, seq in enumerate(ds_split):
        seq_name = seq["sequence_name"]
        seq_dir = os.path.join(out_dir, seq_name)
        os.makedirs(seq_dir, exist_ok=True)

        # Each sample provides an archive path
        # Let's copy/extract the files if provided
        if "archive" in seq:
            archive_path = seq["archive"]
            shutil.unpack_archive(archive_path, seq_dir)

print("✅ DanceTrack download complete.")

