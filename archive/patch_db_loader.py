"""
patch_db_loader.py
Adds spills_per_pipe and hrs_per_pipe to the feat_matrix table definition
in whichever version of db_loader.py is actually on disk.

Run once from the project root:  python patch_db_loader.py
"""

import re

PATH = "src/db_loader.py"

with open(PATH) as f:
    src = f.read()

if "spills_per_pipe" in src:
    print("Already patched - nothing to do.")
    raise SystemExit(0)

# insert the two new columns right after n_overflows in the CREATE TABLE block
pattern = re.compile(r"(\n(\s*)n_overflows\s+INTEGER,)")
match = pattern.search(src)

if not match:
    print("Could not find n_overflows column in feat_matrix definition.")
    print("Patch not applied - check the file manually.")
    raise SystemExit(1)

indent = match.group(2)
addition = (
    f"{match.group(1)}\n"
    f"{indent}spills_per_pipe REAL,\n"
    f"{indent}hrs_per_pipe    REAL,"
)
src = src[:match.start()] + addition + src[match.end():]

with open(PATH, "w") as f:
    f.write(src)

print("Patched successfully.")
print()

# show the result so you can confirm
import subprocess
out = subprocess.run(
    ["grep", "-n", "-A2", "n_overflows     INTEGER,", PATH],
    capture_output=True, text=True
)
print(out.stdout or "(run grep manually to verify)")