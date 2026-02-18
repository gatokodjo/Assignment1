import os
import shutil

# Directory containing your model files
CODE_DIR = "code"

# List of required files
model_files = ["english.model", "danish.model", "swedish.model"]

# Check if each file exists
for model_file in model_files:
    path = os.path.join(CODE_DIR, model_file)
    if os.path.exists(path):
        print(f"{model_file} found.")
    else:
        print(f"{model_file} missing! Please save your trained model as {model_file} in {CODE_DIR}.")

# Optional: package for submission
SUBMISSION_FILE = "submission.zip" 
shutil.make_archive("submission", 'zip', CODE_DIR)
print(f"Submission zip created: {SUBMISSION_FILE}")
