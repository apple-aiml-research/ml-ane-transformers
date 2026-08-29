from pathlib import Path
import shutil

path = Path("ane_transformers/huggingface/distilbert.py")
backup = path.with_name(path.name + ".before_mask_variable_fix")

print(f"[*] Patching {path}")

if not path.exists():
    raise SystemExit("ERROR: distilbert.py not found")

shutil.copy2(path, backup)
print(f"[+] Backup created: {backup}")

text = path.read_text()

# We only want the _ane_forward() implementation.
ane_start = text.find("    def _ane_forward(")
if ane_start == -1:
    raise SystemExit("ERROR: _ane_forward() not found")

ane_end = text.find("\n    def ", ane_start + 10)
if ane_end == -1:
    ane_end = len(text)

block = text[ane_start:ane_end]

# The ANE implementation receives the mask as the argument named `mask`.
#
# My previous compatibility patch incorrectly used `attention_mask`
# internally. Replace those references with `mask`.

if "if attention_mask is not None:" in block:
    block = block.replace(
        "if attention_mask is not None:",
        "if mask is not None:",
    )

if "attention_mask = mask" in block:
    block = block.replace(
        "attention_mask = mask",
        "# Keep the normalized mask in the variable expected by the "
        "original ANE kernel.\n            mask = mask",
    )

# Remove accidental self-assignment if the replacement generated it.
block = block.replace(
    "            mask = mask\n",
    "",
)

# The above replacement can remove the wrong occurrence if there is no
# assignment anymore, so ensure the normalized tensor remains named `mask`.
# We specifically look for the final normalized-mask assignment.
needle = "mask = mask.unsqueeze(2).unsqueeze(3)"

if needle not in block:
    raise SystemExit(
        "ERROR: Could not find the normalized mask conversion. "
        "No changes made to working file."
    )

# Make absolutely sure there is no remaining reference to the nonexistent
# local variable in _ane_forward().
if "attention_mask" in block:
    print("[!] Remaining attention_mask references found inside _ane_forward().")
    print("    They will be changed to mask.")
    block = block.replace("attention_mask", "mask")

text = text[:ane_start] + block + text[ane_end:]

path.write_text(text)

print()
print("SUCCESS: Fixed _ane_forward() mask variable.")
print()
print("The ANE kernel now consistently uses `mask` internally.")
print(f"Patched: {path}")
print(f"Backup:  {backup}")
