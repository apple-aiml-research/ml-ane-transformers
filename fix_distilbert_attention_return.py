from pathlib import Path
import shutil
import re

path = Path("ane_transformers/huggingface/distilbert.py")
backup = path.with_name(path.name + ".before_return_fix")

print(f"[*] Patching {path}")

if not path.exists():
    raise SystemExit("ERROR: distilbert.py not found")

shutil.copy2(path, backup)
print(f"[+] Backup created: {backup}")

text = path.read_text()

start = text.find("    def forward(")
if start == -1:
    raise SystemExit("ERROR: forward() not found")

# Find the MultiHeadSelfAttention forward specifically.
class_start = text.find("class MultiHeadSelfAttention")
if class_start == -1:
    raise SystemExit("ERROR: MultiHeadSelfAttention not found")

start = text.find("    def forward(", class_start)
end = text.find("\n    def ", start + 10)

if start == -1 or end == -1:
    raise SystemExit(
        "ERROR: Could not locate MultiHeadSelfAttention.forward()"
    )

block = text[start:end]

old = """        result = self._ane_forward(
            query,
            key,
            value,
            mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
        )

        attn = result[0]
        attn_weights = result[1]
"""

new = """        result = self._ane_forward(
            query,
            key,
            value,
            mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
        )

        # --------------------------------------------------------------
        # ANE return compatibility
        #
        # The original ANE implementation may return:
        #
        #     (attention_output, attention_weights)
        #
        # while some execution paths return only:
        #
        #     attention_output
        #
        # Hugging Face expects the two-value form.
        # Normalize both cases here.
        # --------------------------------------------------------------

        if isinstance(result, tuple):
            attn = result[0]

            if len(result) > 1:
                attn_weights = result[1]
            else:
                attn_weights = None

        else:
            attn = result
            attn_weights = None
"""

if old not in block:
    print("[!] Expected return block was not found exactly.")

    # Try a more tolerant replacement.
    pattern = re.compile(
        r"""        result = self\._ane_forward\(
            query,
            key,
            value,
            mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
        \)

        attn = result\[0\]
        attn_weights = result\[1\]
""",
        re.MULTILINE,
    )

    if not pattern.search(block):
        raise SystemExit(
            "ERROR: Could not locate the _ane_forward() result handling block.\n"
            "No changes made to the working file."
        )

    block = pattern.sub(new, block, count=1)

else:
    block = block.replace(old, new, 1)

text = text[:start] + block + text[end:]

path.write_text(text)

print()
print("SUCCESS: DistilBERT attention return compatibility installed.")
print()
print("The HF wrapper now accepts:")
print("  (attention_output, attention_weights)")
print("or:")
print("  attention_output")
print()
print(f"Patched: {path}")
print(f"Backup:  {backup}")
