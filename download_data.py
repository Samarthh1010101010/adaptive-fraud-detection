import kagglehub
import os
import shutil

# Download latest version
print("Downloading creditcard fraud dataset from Kaggle...")
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")

print("Path to dataset files:", path)

# Copy creditcard.csv to data/ folder
src_file = os.path.join(path, "creditcard.csv")
dst_file = os.path.join("data", "creditcard.csv")

if os.path.exists(src_file):
    os.makedirs("data", exist_ok=True)
    shutil.copy(src_file, dst_file)
    print(f"✓ Dataset copied to {dst_file}")
else:
    print(f"✗ Dataset file not found at {src_file}")
