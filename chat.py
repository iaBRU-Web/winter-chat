# ============ winter-frost: install + load + chat (with memory) ============
!pip install -q -U transformers accelerate

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "brucoder/winter-frost"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

# Alpaca template this model was instruction-tuned on
PREFIX = ("Below is an instruction that describes a task. "
          "Write a response that appropriately completes the request.\n\n")

MAX_CTX  = 1024
RESERVED = 200
history  = ""

def trim_history():
    global history
    while len(tokenizer(history, add_special_tokens=False)["input_ids"]) > MAX_CTX - RESERVED:
        nxt = history.find("### Instruction:", 1)
        if nxt == -1:
            break
        history = history[nxt:]
#You can change the temperature if you want just to increase the response
def ask(q, max_new_tokens=200, temperature=0.7):
    global history
    prompt = PREFIX + history + f"### Instruction:\n{q}\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    out = model.generate(
        **inputs,
        max_new_tokens = max_new_tokens,
        do_sample = True,
        temperature = temperature,
        top_p = 0.9,
        repetition_penalty = 1.3,
        pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    ans = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    history += f"### Instruction:\n{q}\n\n### Response:\n{ans}\n\n"
    trim_history()
    return ans

print("Type 'reset' to clear, 'exit' to stop.\n")
while True:
    q = input("You: ")
    if q.strip().lower() in ("exit", "quit", ""):
        break
    if q.strip().lower() == "reset":
        history = ""
        print("(cleared)\n")
        continue
    print("\nModel:", ask(q), "\n")
