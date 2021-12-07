import os
import shutil

data_dir = "/home/nghibui/codes/datasets/Project_codenet/data"
dest_dir = "/home/nghibui/codes/datasets/ood_codenet"

for problem in os.listdir(data_dir):
    problem_path = os.path.join(data_dir, problem)
    C_language_dir = os.path.join(problem_path, "C")
    files = os.listdir(C_language_dir)
 
    for fname in files:
        shutil.copy2(os.path.join(C_language_dir,fname), dest_dir)