from pathlib import Path
import shutil
import re

path = Path("ane_transformers/huggingface/distilbert.py")

print("=" * 70)
print("DistilBERT attention repair")
print("=" * 70)

# ------------------------------------------------------------
# Find backups
# ------------------------------------------------------------

backups = [
    path.with_name("distilbert.py.backup3"),
    path.with_name("distilbert.py.before_attention_adapter"),
    path.with_name("distilbert.py.before_layout_fix"),
]

print("\nAvailable backups:")

for b in backups:
    print(f"  {'FOUND' if b.exists() else 'missing'}  {b}")

# Prefer backup3 because that was created before the later
# compatibility patches.
source = None

for b in backups:
    if b.exists():
        source = b
        break

if source is None:
    raise SystemExit(
        "\nERROR: No known backup was found.\n"
        "Do not continue. We need the original distilbert.py."
    )

print(f"\n[+] Using recovery source:")
print(f"    {source}")

# Make a backup of the current state before touching it.
current_backup = path.with_name(
    "distilbert.py.before_repair"
)

if not current_backup.exists():
    shutil.copy2(path, current_backup)
    print(f"[+] Current file backed up to:")
    print(f"    {current_backup}")

# Restore selected backup.
shutil.copy2(source, path)

print(f"[+] Restored {path} from {source}")

text = path.read_text()

# ------------------------------------------------------------
# Locate MultiHeadSelfAttention
# ------------------------------------------------------------

class_start = text.find("class MultiHeadSelfAttention")

if class_start < 0:
    raise SystemExit(
        "ERROR: MultiHeadSelfAttention was not found."
    )

forward_start = text.find(
    "    def forward(",
    class_start
)

if forward_start < 0:
    raise SystemExit(
        "ERROR: MultiHeadSelfAttention.forward() was not found."
    )

# Find end of forward method.
next_def = text.find("\n    def ", forward_start + 10)
next_class = text.find("\nclass ", forward_start + 10)

ends = [x for x in (next_def, next_class) if x >= 0]

if ends:
    forward_end = min(ends)
else:
    forward_end = len(text)

old_forward = text[forward_start:forward_end]

print("\n[+] Original forward discovered")
print(f"    characters: {len(old_forward)}")

# ------------------------------------------------------------
# Verify what we recovered
# ------------------------------------------------------------

print("\n[+] Forward signature preview:")
for line in old_forward.splitlines()[:12]:
    print("    " + line)

# We want the ORIGINAL ANE implementation to have query/key/value.
if not all(x in old_forward for x in [
    "query",
    "key",
    "value",
    "mask",
]):
    raise SystemExit(
        "\nERROR: Selected backup does not contain the expected "
        "original ANE attention implementation.\n\n"
        "The backup selected was:\n"
        f"  {source}\n\n"
        "Run this command and send me the output:\n"
        "  sed -n '50,260p' ane_transformers/huggingface/distilbert.py.backup3"
    )

# ------------------------------------------------------------
# Rename original ANE implementation
# ------------------------------------------------------------

ane_forward = old_forward.replace(
    "    def forward(",
    "    def _ane_forward(",
    1
)

# ------------------------------------------------------------
# Install a SINGLE compatibility wrapper.
#
# IMPORTANT:
#
# The surrounding ANE-optimized DistilBERT implementation is
# already producing:
#
#     [B, D, 1, S]
#
# Therefore the wrapper accepts BOTH:
#
#     [B, S, D]       normal HF
#     [B, D, 1, S]    existing ANE path
#
# This prevents the double-conversion that caused:
#
#     Expected [B,S,D], got [B,D,1,S]
# ------------------------------------------------------------

wrapper = r'''    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            output_attentions=False,
            **kwargs):
        """
        Transformers-compatible front end for the ANE attention kernel.

        Supports both tensor layouts:

            Hugging Face:
                [B, S, D]

            ANE optimized path:
                [B, D, 1, S]
        """

        if hidden_states.dim() == 3:
            # ------------------------------------------------
            # HF layout:
            #
            # [B, S, D]
            #
            # Convert to ANE:
            #
            # [B, D, 1, S]
            # ------------------------------------------------
            query = hidden_states.permute(
                0, 2, 1
            ).unsqueeze(2)

        elif hidden_states.dim() == 4:
            # ------------------------------------------------
            # Already ANE layout.
            #
            # [B, D, 1, S]
            # ------------------------------------------------
            query = hidden_states

        else:
            raise RuntimeError(
                "Unexpected hidden_states rank: "
                f"{hidden_states.dim()}, shape="
                f"{list(hidden_states.shape)}"
            )

        # q/k/v are self-attention, so they use the same tensor.
        key = query
        value = query

        # ----------------------------------------------------
        # Normalize attention mask for the original ANE kernel.
        # ----------------------------------------------------

        mask = attention_mask

        if mask is not None:

            if mask.dim() == 2:
                # [B, S]
                mask = mask.unsqueeze(2).unsqueeze(3)

            elif mask.dim() == 4:
                # Already expanded.
                pass

            else:
                raise RuntimeError(
                    "Unexpected attention_mask shape: "
                    f"{list(mask.shape)}"
                )

        if head_mask is not None:
            raise NotImplementedError(
                "head_mask is not supported by the ANE "
                "attention implementation"
            )

        # ----------------------------------------------------
        # Original ANE attention implementation.
        # ----------------------------------------------------

        result = self._ane_forward(
            query,
            key,
            value,
            mask,
            head_mask=None,
            output_attentions=output_attentions,
        )

        attn_output = result[0]
        attn_weights = result[1]

        # ----------------------------------------------------
        # The original ANE implementation returns:
        #
        #     [B, D, 1, S]
        #
        # The surrounding ANE-optimized model expects that
        # layout, so DO NOT blindly convert 4D output to HF.
        #
        # However, if the kernel returns [B,S,D], leave it
        # alone.
        # ----------------------------------------------------

        if attn_output.dim() == 4:
            if (
                attn_output.shape[1] == self.dim
                and attn_output.shape[2] == 1
            ):
                # Already ANE layout.
                return attn_output, attn_weights

        elif attn_output.dim() == 3:
            # Normal HF layout.
            return attn_output, attn_weights

        raise RuntimeError(
            "Unexpected attention output shape: "
            f"{list(attn_output.shape)}"
        )

'''

new_forward = wrapper + ane_forward

text = (
    text[:forward_start]
    + new_forward
    + text[forward_end:]
)

path.write_text(text)

print("\n[+] Installed SINGLE attention compatibility layer")

# ------------------------------------------------------------
# Syntax validation
# ------------------------------------------------------------

compile(
    path.read_text(),
    str(path),
    "exec"
)

print("[+] Python syntax check: PASS")

print("\n" + "=" * 70)
print("REPAIR COMPLETE")
print("=" * 70)

print("\nCurrent file:")
print(f"  {path}")

print("\nRecovery source:")
print(f"  {source}")

print("\nCurrent-state backup:")
print(f"  {current_backup}")

print("\nNext command:")
print(
    "  python3 -m unittest "
    "ane_transformers.huggingface.test_distilbert -v"
)
