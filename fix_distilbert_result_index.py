from pathlib import Path
import shutil
import re

path = Path("ane_transformers/huggingface/distilbert.py")
backup = path.with_name(path.name + ".before_result_index_fix")

print(f"[*] Inspecting {path}")

if not backup.exists():
    shutil.copy2(path, backup)
    print(f"[+] Backup created: {backup}")
else:
    print(f"[=] Backup already exists: {backup}")

text = path.read_text()

# Find the actual unsafe result[1] access.
matches = list(re.finditer(r'(?m)^(\s*)attn_weights\s*=\s*result\[1\]\s*$', text))

if not matches:
    print("[!] Could not find the exact unsafe:")
    print("    attn_weights = result[1]")
    print()
    print("[*] Current result references:")
    for n, line in enumerate(text.splitlines(), 1):
        if "result[" in line or "attn_weights" in line:
            print(f"{n}: {line}")
    raise SystemExit(1)

m = matches[0]
indent = m.group(1)

replacement = f'''{indent}# ANE attention can return either a tensor or a tuple.
{indent}# Hugging Face expects (attention_output, attention_weights).
{indent}if isinstance(result, (tuple, list)):
{indent}    attn_output = result[0]
{indent}    attn_weights = result[1] if len(result) > 1 else None
{indent}else:
{indent}    attn_output = result
{indent}    attn_weights = None'''

text = text[:m.start()] + replacement + text[m.end():]

# The old code may subsequently return/use result[0].
# Replace only nearby return expressions involving result[0].
lines = text.splitlines()
for i, line in enumerate(lines):
    if "return result[0]" in line:
        lines[i] = line.replace("result[0]", "attn_output")

text = "\n".join(lines) + "\n"
path.write_text(text)

print("[+] Removed unsafe result[1] access")
print("[+] Added one-/two-value ANE result handling")
print("[+] Converted nearby result[0] return to attn_output")
print()
print("SUCCESS: DistilBERT result handling fixed.")
print(f"Patched: {path}")
print(f"Backup:  {backup}")
