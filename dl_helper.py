import os, sys, subprocess, zipfile, csv, urllib.request
from pathlib import Path

BASE = Path(os.path.join(os.sep, "Users", "virtu", "Documents", "SkinTag", "data"))
print("BASE:", BASE)
print("exists:", BASE.exists())