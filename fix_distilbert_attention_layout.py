from pathlib import Path
import re
import shutil

path = Path("ane_transformers/huggingface/distilbert.py")

if not path.exists():
    raise SystemExit(f"ERROR: {path} not found")

backup = path.with_suffix(".py.before_layout_fix")

if not backup.exists():
    shutil.copy2(path, backup)
    print(f"[+] Backup created: {backup}")
else:
    print(f"[=] Backup already exists: {backup}")

text = path.read_text()

# ------------------------------------------------------------
# Find MultiHeadSelfAttention.forward()
# ------------------------------------------------------------

class_start = text.find("class MultiHeadSelfAttention")
if class_start < 0:
    raise SystemExit("ERROR: MultiHeadSelfAttention class not found")

forward_start = text.find("    def forward(", class_start)
if forward_start < 0:
    raise SystemExit("ERROR: MultiHeadSelfAttention.forward() not found")

# Find the next method/class after forward.
next_def = text.find("\n    def ", forward_start + 10)
next_class = text.find("\nclass ", forward_start + 10)

ends = [x for x in (next_def, next_class) if x >= 0]

if not ends:
    forward_end = len(text)
else:
    forward_end = min(ends)

old_forward = text[forward_start:forward_end]

print("[+] Found MultiHeadSelfAttention.forward()")
print(f"[+] Existing forward block: {len(old_forward)} characters")

# ------------------------------------------------------------
# We preserve the existing ANE implementation.
#
# The important change is that the public HF-facing forward()
# accepts:
#
#   hidden_states: [B, S, D]
#
# and converts it to:
#
#   [B, D, 1, S]
#
# before invoking the existing ANE attention implementation.
# ------------------------------------------------------------

# Extract the existing ANE body after the docstring/comments
# by replacing the existing method with a wrapper that calls
# a private implementation containing the original body.

# First determine whether we've already installed this fix.
if "_ane_forward" in old_forward:
    raise SystemExit(
        "ERROR: _ane_forward already exists in this forward(). "
        "The layout adapter may already be installed."
    )

# Extract the original argument/body after the function declaration.
# We will rename the original implementation to _ane_forward.
ane_forward = old_forward.replace(
    "    def forward(",
    "    def _ane_forward(",
    1
)

# The original ANE implementation uses query/key/value/mask.
# We deliberately leave that implementation alone.
#
# Insert a new HF-compatible forward before it.

wrapper = r'''    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            output_attentions=False,
            **kwargs):
        """
        Hugging Face-compatible wrapper around the ANE attention kernel.

        Hugging Face DistilBERT:
            hidden_states = [batch, sequence, hidden]

        ANE implementation:
            [batch, hidden, 1, sequence]

        We perform the layout conversion here so the original ANE
        implementation does not need to know about the newer
        Transformers calling convention.
        """

        if hidden_states.dim() != 3:
            raise RuntimeError(
                "Expected hidden_states with shape "
                "[batch, sequence, hidden], got "
                f"{list(hidden_states.shape)}"
            )

        batch, sequence, hidden = hidden_states.shape

        if hidden != self.dim:
            raise RuntimeError(
                f"Expected hidden dimension {self.dim}, got {hidden}"
            )

        # --------------------------------------------------------
        # HF -> ANE
        #
        # [B, S, D]
        #      |
        #      v
        # [B, D, 1, S]
        # --------------------------------------------------------

        query = hidden_states.permute(0, 2, 1).unsqueeze(2)
        key = query
        value = query

        # --------------------------------------------------------
        # Normalize the Hugging Face attention mask.
        #
        # HF normally supplies:
        #
        #   [B, S]
        #
        # ANE implementation expects:
        #
        #   [B, S, 1, 1]
        # --------------------------------------------------------

        mask = attention_mask

        if mask is not None:
            if mask.dim() == 2:
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
                "head_mask is not supported by the ANE attention implementation"
            )

        # --------------------------------------------------------
        # Run original ANE attention.
        # --------------------------------------------------------

        attn_output, attn_weights = self._ane_forward(
            query,
            key,
            value,
            mask,
            head_mask=None,
            output_attentions=output_attentions,
        )

        # --------------------------------------------------------
        # ANE -> HF
        #
        # [B, D, 1, S]
        #      |
        #      v
        # [B, S, D]
        # --------------------------------------------------------

        if attn_output.dim() != 4:
            raise RuntimeError(
                "ANE attention returned unexpected shape: "
                f"{list(attn_output.shape)}"
            )

        attn_output = (
            attn_output
            .squeeze(2)
            .permute(0, 2, 1)
            .contiguous()
        )

        if attn_output.shape != (batch, sequence, hidden):
            raise RuntimeError(
                "ANE -> HF attention conversion produced unexpected "
                f"shape {list(attn_output.shape)}, expected "
                f"[{batch}, {sequence}, {hidden}]"
            )

        if output_attentions:
            return attn_output, attn_weights

        return attn_output, None

'''

# ------------------------------------------------------------
# Replace original forward with wrapper + renamed implementation
# ------------------------------------------------------------

new_forward = wrapper + ane_forward

text = text[:forward_start] + new_forward + text[forward_end:]

path.write_text(text)

print("[+] Installed HF -> ANE -> HF attention layout adapter")
print()
print("SUCCESS")
print(f"Patched: {path}")
print(f"Backup:  {backup}")

