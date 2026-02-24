import os
import shutil

p = r"d:\malware_entropy_ml\entropy_ids\controller"
if os.path.exists(p):
    shutil.rmtree(p)
    print("Deleted obsolete controller")
