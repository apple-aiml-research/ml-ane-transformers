from pathlib import Path
import shutil
import re

path = Path("ane_transformers/huggingface/distilbert.py")
backup = path.with_name(path.name + ".before_mask_fix")

print(f"[*] Patching {path}")

if not path.exists():
    raise SystemExit(f"ERROR: {path} not found")

shutil.copy2(path, backup)
print(f"[+] Backup created: {backup}")

text = path.read_text()

# Locate the mask-validation block inside _ane_forward().
start_marker = "        # Validate mask"
end_marker = "        if head_mask is not None:"

start = text.find(start_marker)
end = text.find(end_marker, start)

if start == -1 or end == -1:
    raise SystemExit(
        "ERROR: Could not locate the ANE mask-validation block.\n"
        "No changes were made to the working file."
    )

old_block = text[start:end]

new_block = r'''        # --------------------------------------------------------------
        # Normalize Hugging Face attention masks for the legacy ANE kernel.
        #
        # The original ANE implementation expects:
        #
        #     [batch, sequence, 1, 1]
        #
        # Modern Transformers may provide:
        #
        #     [batch, sequence]
        #     [batch, 1, 1, sequence]
        #     [batch, 1, hidden, sequence]
        #
        # The last form is what the current Transformers stack is producing
        # here: [2, 1, 768, 256].
        #
        # The 768 dimension is NOT the sequence dimension.  The sequence
        # dimension is the final dimension (256).
        # --------------------------------------------------------------

        if attention_mask is not None:

            mask = attention_mask

            if mask.dim() == 2:
                # [B, S]
                pass

            elif mask.dim() == 4:
                # Modern HF expanded mask.
                #
                # Examples:
                #   [B, 1, 1, S]
                #   [B, 1, H, S]
                #
                # In both cases the final dimension is the key sequence.
                #
                # Take the first query/head row. For the current DistilBERT
                # path those rows contain the same key-position mask.
                if mask.size(0) != bs:
                    raise RuntimeError(
                        "Attention mask batch dimension does not match "
                        f"hidden states: mask={list(mask.size())}, batch={bs}"
                    )

                if mask.size(-1) != seqlen:
                    raise RuntimeError(
                        "Attention mask sequence dimension does not match "
                        f"hidden states: mask={list(mask.size())}, "
                        f"expected sequence={seqlen}"
                    )

                mask = mask[:, 0, 0, :]

            elif mask.dim() == 3:
                # Be permissive with [B, 1, S] or [B, S, 1].
                if mask.size(0) != bs:
                    raise RuntimeError(
                        "Attention mask batch dimension does not match "
                        f"hidden states: mask={list(mask.size())}, batch={bs}"
                    )

                if mask.size(-1) == seqlen:
                    mask = mask[:, 0, :]
                elif mask.size(1) == seqlen:
                    mask = mask[:, :, 0]
                else:
                    raise RuntimeError(
                        "Unsupported 3-D attention mask shape: "
                        f"{list(mask.size())}; expected sequence={seqlen}"
                    )

            else:
                raise RuntimeError(
                    "Unsupported attention mask rank: "
                    f"{mask.dim()}, shape={list(mask.size())}"
                )

            # At this point mask should be [B, S].
            if list(mask.size()) != [bs, seqlen]:
                raise RuntimeError(
                    "Failed to normalize attention mask: "
                    f"got {list(mask.size())}, expected {[bs, seqlen]}"
                )

            # Normalize dtype/semantics.
            #
            # Boolean:
            #   True  = token is valid
            #   False = token is padding
            #
            # Integer 0/1:
            #   1 = valid
            #   0 = padding
            #
            # Floating-point masks can already be additive masks
            # (0 for valid, negative value for masked), which should
            # be preserved.
            if mask.dtype == torch.bool:
                mask = mask.logical_not().to(dtype=torch.float32) * -1e4

            elif mask.dtype in (
                torch.int8,
                torch.uint8,
                torch.int16,
                torch.int32,
                torch.int64,
            ):
                mask = (1 - mask.to(dtype=torch.float32)) * -1e4

            elif not mask.is_floating_point():
                raise TypeError(
                    f"Unexpected dtype for mask: {mask.dtype}"
                )

            else:
                mask = mask.to(dtype=torch.float32)

                # If this is a normal binary floating-point mask,
                # convert 0/1 -> additive 0/-1e4.
                #
                # If it is already an additive mask containing negative
                # values, leave it alone.
                if bool(torch.all((mask == 0) | (mask == 1)).item()):
                    mask = (1 - mask) * -1e4

            # Legacy ANE layout:
            #
            # [B, S] -> [B, S, 1, 1]
            mask = mask.unsqueeze(2).unsqueeze(3)

            expected_mask_shape = [bs, seqlen, 1, 1]

            if list(mask.size()) != expected_mask_shape:
                raise RuntimeError(
                    "Invalid normalized mask shape "
                    f"(Expected {expected_mask_shape}, "
                    f"got {list(mask.size())})"
                )

            # The rest of the original ANE kernel expects the variable
            # to be named `mask`.
            attention_mask = mask

        else:
            mask = None

'''

print("[+] Replacing existing mask-validation block")
text = text[:start] + new_block + text[end:]

path.write_text(text)

print()
print("SUCCESS: DistilBERT attention-mask compatibility patch installed.")
print()
print(f"Patched: {path}")
print(f"Backup:  {backup}")
