import time, math, random, torch
import bitsandbytes as bnb
from datasets import load_dataset
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
from huggingface_hub import HfApi, hf_hub_download, create_repo

assert torch.cuda.is_available(), "Switch runtime to T4 GPU!"
REPO = "brucoder/winter-frost-700m"        # checkpoints live here
SEQ, MB, ACC = 1024, 4, 16                 # 65,536 tokens per step
TOK_PER_STEP = SEQ*MB*ACC
PEAK, WARM = 3e-4, 1000
DECAY_AT, DECAY_STEPS, MIN_LR = None, 3000, 3e-5   # set DECAY_AT=<step> to finish
SESSION_MIN, SAVE_MIN = 600, 30
TARGET_TOKENS = 1_000_000_000_000

tok = AutoTokenizer.from_pretrained("brucoder/winter-frost")   # your 32k tokenizer
eos = tok.eos_token_id if tok.eos_token_id is not None else tok.pad_token_id
api = HfApi(); create_repo(REPO, exist_ok=True)

# ---------- build ~700M model or resume ----------
step, docs, state = 0, 0, None
try:
    state = torch.load(hf_hub_download(REPO, "state.pt"), map_location="cpu")
    model = LlamaForCausalLM.from_pretrained(REPO, torch_dtype=torch.float32)
    step, docs = state["step"], state["docs"]
    print(f"Resuming at step {step} ({step*TOK_PER_STEP/1e9:.2f}B tokens)")
except Exception:
    cfg = LlamaConfig(vocab_size=len(tok), hidden_size=1536, intermediate_size=4096,
                      num_hidden_layers=23, num_attention_heads=12, num_key_value_heads=12,
                      max_position_embeddings=SEQ, tie_word_embeddings=True,
                      bos_token_id=eos, eos_token_id=eos)
    model = LlamaForCausalLM(cfg)
    print("Built fresh model:", round(sum(p.numel() for p in model.parameters())/1e6), "M params")
model.gradient_checkpointing_enable()
model.config.use_cache = False
model.to("cuda").train()

decay = [p for p in model.parameters() if p.ndim >= 2]
nodec = [p for p in model.parameters() if p.ndim < 2]
opt = bnb.optim.AdamW8bit([{"params": decay, "weight_decay": 0.1},
                           {"params": nodec, "weight_decay": 0.0}],
                          lr=PEAK, betas=(0.9, 0.95))
scaler = torch.amp.GradScaler("cuda")
if state:
    opt.load_state_dict(state["opt"]); scaler.load_state_dict(state["scaler"])

# ---------- streaming packed data ----------
def batches():
    global docs
    ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-100BT", split="train",
                      streaming=True).skip(docs)
    buf, need = [], SEQ*MB
    for ex in ds:
        docs += 1
        buf += tok(ex["text"], add_special_tokens=False)["input_ids"] + [eos]
        while len(buf) >= need:
            yield torch.tensor(buf[:need]).view(MB, SEQ); buf = buf[need:]
it = batches()

def save():
    model.save_pretrained("ckpt"); tok.save_pretrained("ckpt")
    torch.save({"opt": opt.state_dict(), "scaler": scaler.state_dict(),
                "step": step, "docs": docs}, "ckpt/state.pt")
    api.upload_folder(folder_path="ckpt", repo_id=REPO)
    print(f"  saved checkpoint at step {step}")

def get_lr(s):
    if s < WARM: return PEAK*(s+1)/WARM
    if DECAY_AT is None or s < DECAY_AT: return PEAK
    f = min((s-DECAY_AT)/DECAY_STEPS, 1.0)
    return PEAK*(1-f) + MIN_LR*f

# ---------- train ----------
t0 = tl = ts = time.time()
while True:
    if DECAY_AT is not None and step >= DECAY_AT + DECAY_STEPS: print("Decay finished."); break
    lr = get_lr(step)
    for g in opt.param_groups: g["lr"] = lr
    for _ in range(ACC):
        x = next(it).cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = model(input_ids=x, labels=x).loss / ACC
        scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
    step += 1
    if step % 5 == 0:
        tps = 5*TOK_PER_STEP/(time.time()-tl); tl = time.time()
        print(f"step {step} | loss {loss.item()*ACC:.3f} | {tps:,.0f} tok/s | "
              f"{step*TOK_PER_STEP/1e9:.3f}B / {TARGET_TOKENS/1e9:.0f}B tokens")
    if time.time()-ts > SAVE_MIN*60: save(); ts = time.time()
    if time.time()-t0 > SESSION_MIN*60: break
save()
print("Session ended. Rerun this cell to resume.")
