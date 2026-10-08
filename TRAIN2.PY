import torch, random, time, math
from itertools import islice
from datasets import load_dataset, Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback,
                          TrainingArguments, DataCollatorForSeq2Seq)

assert torch.cuda.is_available(), "Switch runtime to T4 GPU first!"
BASE, OUT = "brucoder/winter-frost-1-extra-v2", "brucoder/winter-frost-1-extra-v3"
SKIP, TAKE = 10000, 10000          # next session: SKIP = 20000
MAXLEN, TRAIN_MIN, PEAK = 256, 6, 2e-5   # next sessions: 1.5e-5, then 1e-5
HEAD = ("Below is an instruction that describes a task. Write a response that "
        "appropriately completes the request.\n\n")
TURN = "### Instruction:\n{}\n\n### Response:\n"

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE).to("cuda")
eos = tok.eos_token_id if tok.eos_token_id is not None else tok.pad_token_id
if tok.pad_token_id is None: tok.pad_token_id = eos
model.config.pad_token_id = tok.pad_token_id
head_ids = tok(HEAD)["input_ids"]

def to_turns(msgs):
    turns, u = [], None
    for m in msgs:
        if m["role"] == "user": u = m["content"]
        elif m["role"] == "assistant" and u is not None:
            turns.append((u, m["content"])); u = None
    return turns

def encode_conv(turns):
    ids = list(head_ids); labels = [-100]*len(ids); kept = 0
    for u, a in turns:
        p = tok(TURN.format(u), add_special_tokens=False)["input_ids"]
        r = tok(a, add_special_tokens=False)["input_ids"] + [eos]
        if len(ids)+len(p)+len(r) > MAXLEN: break
        ids += p + r; labels += [-100]*len(p) + r; kept += 1
    return (ids, labels) if kept else None

# ---------- data: fresh UltraChat + everyday chat + a little Alpaca + identity ----------
convs = [to_turns(ex["messages"]) for ex in
         load_dataset("HuggingFaceTB/smoltalk", "everyday-conversations", split="train")]
s = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True).skip(SKIP)
convs += [to_turns(ex["messages"]) for ex in islice(s, TAKE)]
alp = load_dataset("yahma/alpaca-cleaned", split="train").shuffle(seed=2).select(range(2000))
convs += [[(ex["instruction"] + ("\n" + ex["input"] if ex["input"] else ""), ex["output"])]
          for ex in alp]
identity = [
    ("Who are you?", "I'm Winter-Frost, a small language model built by brucoder."),
    ("Who made you?", "I was created by brucoder as a from-scratch learning project."),
    ("Are you an AI?", "Yes, I'm an AI language model called Winter-Frost."),
    ("What is your name?", "My name is Winter-Frost."),
    ("Can you help me?", "Of course! What do you need help with?"),
    ("help pls", "Sure, I'm happy to help. What's the problem?"),
]
convs += [[t] for t in identity] * 30

rows = []
for t in convs:
    e = encode_conv(t) if t else None
    if e: rows.append({"input_ids": e[0], "attention_mask": [1]*len(e[0]), "labels": e[1]})
random.seed(0); random.shuffle(rows)
ds = Dataset.from_list(rows)
print("training conversations:", len(ds))

# ---------- time-based cosine LR + hard stop ----------
class TimeCosine(TrainerCallback):
    def on_train_begin(self, args, state, control, **kw): self.t0 = time.time()
    def on_step_begin(self, args, state, control, **kw):
        f = min((time.time()-self.t0)/(TRAIN_MIN*60), 1.0)
        lr = PEAK*(f/0.05 if f < 0.05 else 0.05 + 0.95*0.5*(1+math.cos(math.pi*f)))
        opt = kw.get("optimizer")
        if opt is not None:
            for g in opt.param_groups: g["lr"] = lr
    def on_step_end(self, args, state, control, **kw):
        if time.time()-self.t0 > TRAIN_MIN*60: control.should_training_stop = True

Trainer(model=model, train_dataset=ds, callbacks=[TimeCosine()],
    data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100,
                                         pad_to_multiple_of=8),
    args=TrainingArguments(output_dir="out", num_train_epochs=2, learning_rate=PEAK,
        lr_scheduler_type="constant", per_device_train_batch_size=32,
        weight_decay=0.01, fp16=True, dataloader_num_workers=2,
        logging_steps=50, save_strategy="no", report_to="none")).train()

# ---------- quick test ----------
model.eval()
for q in ["Hi!", "Who are you?", "What is a computer?"]:
    ids = head_ids + tok(TURN.format(q), add_special_tokens=False)["input_ids"]
    x = torch.tensor([ids]).to(model.device)
    with torch.no_grad():
        out = model.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=80,
            do_sample=True, temperature=0.5, top_p=0.9, repetition_penalty=1.2,
            eos_token_id=eos, pad_token_id=tok.pad_token_id)
    print(f">>> {q}\n{tok.decode(out[0][len(ids):], skip_special_tokens=True).strip()}\n")

# ---------- upload (new name, so v2 stays safe) ----------
model.push_to_hub(OUT); tok.push_to_hub(OUT)
print("Uploaded: https://huggingface.co/" + OUT)
