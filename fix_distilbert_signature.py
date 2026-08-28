from pathlib import Path
import shutil
import re
import inspect
import sys

FILE = Path("ane_transformers/huggingface/distilbert.py")
BACKUP = Path("ane_transformers/huggingface/distilbert.py.backup4")

OLD = """def forward(self,
                query,
                key,
                value,
                mask,
                head_mask=None,
                output_attentions=False):"""

NEW = """def forward(self,
                hidden_states,
                attention_mask=None,
                head_mask=None,
                output_attentions=False,
                **kwargs):"""

if not FILE.exists():
    print(f"ERROR: {FILE} does not exist.")
    sys.exit(1)

text = FILE.read_text()

if OLD not in text:
    print("ERROR: Expected old forward() signature was not found.")
    print()
    print("Searching for the current MultiHeadSelfAttention.forward():")
    
    match = re.search(
        r"class MultiHeadSelfAttention.*?(?=\\nclass |\\Z)",
        text,
        re.DOTALL,
    )

    if match:
        block = match.group(0)
        sig = re.search(
            r"    def forward\\(.*?\\):",
            block,
            re.DOTALL,
        )
        if sig:
            print(sig.group(0))
    else:
        print("Could not locate MultiHeadSelfAttention.")

    sys.exit(2)

# Don't overwrite an existing backup.
if not BACKUP.exists():
    shutil.copy2(FILE, BACKUP)
    print(f"Backup created: {BACKUP}")
else:
    print(f"Backup already exists: {BACKUP}")

# Apply ONLY the signature change.
text = text.replace(OLD, NEW, 1)
FILE.write_text(text)

print(f"Patched: {FILE}")
print()
print("New signature written:")
print(NEW)

print()
print("Verifying imported module...")

# Make sure the local repository is first on sys.path.
sys.path.insert(0, str(Path.cwd()))

from ane_transformers.huggingface import distilbert

print()
print("Imported file:")
print(distilbert.__file__)

print()
print("Python sees this signature:")
print(inspect.signature(distilbert.MultiHeadSelfAttention.forward))

expected = "(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False, **kwargs)"

actual = str(inspect.signature(distilbert.MultiHeadSelfAttention.forward))

if actual == expected:
    print()
    print("SUCCESS: DistilBERT attention signature is now Transformers-compatible.")
else:
    print()
    print("WARNING: Signature differs from the expected form.")
    print("Expected:", expected)
    print("Actual:  ", actual)
