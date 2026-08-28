from pathlib import Path
import re
import shutil

path = Path("ane_transformers/huggingface/distilbert.py")
backup = path.with_suffix(".py.before_attention_adapter")

print(f"[*] Patching {path}")

if not path.exists():
    raise SystemExit(f"ERROR: {path} does not exist")

shutil.copy2(path, backup)
print(f"[+] Backup created: {backup}")

text = path.read_text()

# ------------------------------------------------------------------
# 1. Make sure we're inheriting from the Transformers 5.x class.
# ------------------------------------------------------------------

old_base = "class MultiHeadSelfAttention(modeling_distilbert.MultiHeadSelfAttention):"
new_base = "class MultiHeadSelfAttention(modeling_distilbert.DistilBertSelfAttention):"

if old_base in text:
    text = text.replace(old_base, new_base)
    print("[+] Fixed MultiHeadSelfAttention base class")
elif new_base in text:
    print("[=] Base class already correct")
else:
    raise SystemExit("ERROR: Could not find MultiHeadSelfAttention class")

# ------------------------------------------------------------------
# 2. Replace the forward() signature.
#
# Transformers 5.x calls:
#
#   attention(
#       hidden_states,
#       attention_mask=attention_mask,
#       ...
#   )
#
# Apple's old implementation expected:
#
#   query, key, value, mask
# ------------------------------------------------------------------

old_signature = """    def forward(self,
                query,
                key,
                value,
                mask,
                head_mask=None,
                output_attentions=False):"""

new_signature = """    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            output_attentions=False,
            **kwargs):"""

if old_signature not in text:
    raise SystemExit(
        "ERROR: Expected old forward() signature was not found. "
        "Your file may already have been modified differently."
    )

text = text.replace(old_signature, new_signature, 1)

print("[+] Replaced old attention signature")

# ------------------------------------------------------------------
# 3. Insert the adapter immediately before the old Apple attention body.
#
# HF:
#       hidden_states = [batch, sequence, dim]
#
# Apple ANE implementation:
#       [batch, dim, 1, sequence]
#
# Self-attention means Q/K/V all originate from hidden_states.
# ------------------------------------------------------------------

needle = """        # Parse tensor shapes for source and target sequences
        assert len(query.size()) == 4 and len(key.size()) == 4 and len(
            value.size()) == 4
"""

adapter = """        # --------------------------------------------------------------
        # Transformers 5.x -> Apple ANE tensor-layout adapter
        #
        # Hugging Face supplies:
        #     hidden_states = [batch, sequence, hidden_size]
        #
        # The original Apple ANE implementation operates on:
        #     [batch, hidden_size, 1, sequence]
        #
        # DistilBERT self-attention uses the same tensor for Q/K/V.
        # --------------------------------------------------------------

        if hidden_states.dim() != 3:
            raise RuntimeError(
                f"Expected hidden_states to be 3-D "
                f"[batch, sequence, dim], got {list(hidden_states.shape)}"
            )

        query = hidden_states.permute(0, 2, 1).unsqueeze(2)
        key = query
        value = query

        mask = attention_mask

        # The current Transformers attention mask normally arrives as:
        #     [batch, sequence]
        #
        # The original Apple implementation expects:
        #     [batch, sequence, 1, 1]
        if mask is not None and mask.dim() == 2:
            mask = mask.unsqueeze(2).unsqueeze(3)

        # --------------------------------------------------------------
        # Original Apple ANE implementation starts here.
        # --------------------------------------------------------------

        # Parse tensor shapes for source and target sequences
        assert len(query.size()) == 4 and len(key.size()) == 4 and len(
            value.size()) == 4
"""

text = text.replace(needle, adapter, 1)

print("[+] Added HF -> ANE tensor adapter")

# ------------------------------------------------------------------
# 4. The original method returns:
#
#       attn, attn_weights
#
# where attn is [batch, dim, 1, sequence].
#
# Transformers expects:
#
#       [batch, sequence, dim]
#
# Find the final ANE output section and convert it back.
# ------------------------------------------------------------------

old_return_area = """        attn = self.out_lin(attn)
"""

if old_return_area not in text:
    raise SystemExit(
        "ERROR: Could not find 'attn = self.out_lin(attn)'."
    )

new_return_area = """        attn = self.out_lin(attn)

        # --------------------------------------------------------------
        # Apple ANE -> Transformers 5.x tensor-layout adapter
        #
        # ANE:
        #     [batch, dim, 1, sequence]
        #
        # Transformers:
        #     [batch, sequence, dim]
        # --------------------------------------------------------------

        attn = attn.squeeze(2).permute(0, 2, 1).contiguous()

        if output_attentions:
            # Preserve the attention weights when requested.
            return attn, torch.stack(attn_weights, dim=1).squeeze(2)

        return attn, None
"""

text = text.replace(old_return_area, new_return_area, 1)

print("[+] Added ANE -> HF output adapter")

path.write_text(text)

print()
print("SUCCESS: DistilBERT attention adapter installed.")
print()
print(f"Backup: {backup}")
print(f"Patched: {path}")
