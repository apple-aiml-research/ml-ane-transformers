from pathlib import Path
import shutil

path = Path("ane_transformers/huggingface/distilbert.py")
backup = path.with_name(path.name + ".before_result_tuple_fix")

print(f"[*] Patching {path}")

if not backup.exists():
    shutil.copy2(path, backup)
    print(f"[+] Backup created: {backup}")
else:
    print(f"[=] Backup already exists: {backup}")

text = path.read_text()

old = """        attn_weights = result[1]
        if output_attentions:
            return result[0], attn_weights
        return result[0], None
"""

new = """        # The ANE implementation may return either:
        #   (attention_output, attention_weights)
        # or just:
        #   attention_output
        #
        # Hugging Face always expects the wrapper to return:
        #   (attention_output, attention_weights)

        if isinstance(result, (tuple, list)):
            attn_output = result[0]

            if len(result) > 1:
                attn_weights = result[1]
            else:
                attn_weights = None
        else:
            attn_output = result
            attn_weights = None

        if output_attentions:
            return attn_output, attn_weights

        return attn_output, None
"""

if old not in text:
    print("[!] Exact old result-handling block was not found.")
    print("[!] Showing nearby result[1] references:")
    for i, line in enumerate(text.splitlines(), 1):
        if "result[1]" in line or "attn_weights = result" in line:
            print(f"{i}: {line}")
    raise SystemExit(1)

text = text.replace(old, new, 1)

path.write_text(text)

print("[+] Replaced unsafe result[1] access")
print("[+] ANE result handling now supports 1- or 2-element returns")
print()
print("SUCCESS")
print(f"Patched: {path}")
print(f"Backup:  {backup}")
