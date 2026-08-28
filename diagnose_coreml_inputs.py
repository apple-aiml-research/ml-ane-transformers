from pathlib import Path

p = Path("ane_transformers/huggingface/test_distilbert.py")
text = p.read_text()

print("=== CoreML conversion section ===")

lines = text.splitlines()

for i, line in enumerate(lines, 1):
    if "torch.jit.trace" in line or "ct.convert" in line or "inputs=" in line:
        start = max(1, i - 8)
        end = min(len(lines), i + 15)

        print()
        print(f"--- around line {i} ---")

        for n in range(start, end + 1):
            print(f"{n:4}: {lines[n-1]}")
