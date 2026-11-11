import os
import struct
import subprocess
import threading
import json
import time
from typing import Optional
import psutil
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel
from huggingface_hub import hf_hub_download
import tiktoken

REPO_ID       = "NOT-OMEGA/OmegaMind"
MODEL_BIN     = "model.bin"
TOK_BIN       = "tokenizer.bin"
EXEC_BIN      = "./inference"
SYSTEM_PROMPT = "System: You are OmegaMind, a helpful, thoughtful, and articulate AI assistant.\n"

app = FastAPI(title="OmegaMind AVX2 Inference Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

proc: Optional[subprocess.Popen] = None
proc_lock = threading.Lock()
vocab_bytes: dict[int, bytes] = {}
enc = tiktoken.get_encoding("gpt2")
EOT = enc.eot_token

psutil.cpu_percent(interval=None)


def get_system_metrics():
    """Returns live CPU % and RAM % usage."""
    cpu_pct = round(psutil.cpu_percent(interval=None), 1)
    vm = psutil.virtual_memory()
    ram_pct = round(vm.percent, 1)
    ram_used_gb = round(vm.used / (1024 ** 3), 2)
    ram_total_gb = round(vm.total / (1024 ** 3), 2)
    return {
        "cpu_pct": cpu_pct,
        "ram_pct": ram_pct,
        "ram_used_gb": ram_used_gb,
        "ram_total_gb": ram_total_gb,
    }


def ensure_artifacts():
    """Downloads model.bin & tokenizer.bin from NOT-OMEGA/OmegaMind and compiles inference.cpp."""
    for fname in [TOK_BIN, MODEL_BIN]:
        if not os.path.exists(fname):
            print(f"[OmegaMind] Downloading {fname} from {REPO_ID}...")
            hf_hub_download(
                repo_id=REPO_ID,
                filename=fname,
                local_dir=".",
                token=os.environ.get("HF_TOKEN")
            )
            print(f"[OmegaMind] ✅ {fname} ready ({os.path.getsize(fname)/1e6:.2f} MB)")

    with open(TOK_BIN, "rb") as f:
        (vocab_size,) = struct.unpack("i", f.read(4))
        for i in range(vocab_size):
            (blen,) = struct.unpack("i", f.read(4))
            vocab_bytes[i] = f.read(blen)
    print(f"[OmegaMind] ✅ Loaded {len(vocab_bytes):,} tokens from {TOK_BIN}")

    if not os.path.exists(EXEC_BIN):
        print("[OmegaMind] Compiling inference.cpp with AVX2 + FMA + OpenMP...")
        cmd = ["g++", "-O3", "-mavx2", "-mfma", "-fopenmp", "-std=c++17", "inference.cpp", "-o", "inference"]
        subprocess.run(cmd, check=True)
        print("[OmegaMind] ✅ Compiled ./inference binary!")


def start_engine():
    global proc
    ensure_artifacts()
    proc = subprocess.Popen(
        [EXEC_BIN],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    ready_line = proc.stdout.readline().strip()
    if ready_line != "READY":
        err = proc.stderr.read()
        raise RuntimeError(f"C++ Engine failed to start: {ready_line} | {err}")
    print("[OmegaMind] 🚀 C++17 AVX2 Engine is READY!")


@app.on_event("startup")
def on_startup():
    start_engine()


class ChatRequest(BaseModel):
    session_id: str = "default"
    message: str
    max_tokens: int = 256
    temperature: float = 0.25
    top_k: int = 8
    top_p: float = 0.85
    repetition_penalty: float = 1.18
    is_first_turn: bool = False


class ResetRequest(BaseModel):
    session_id: str = "default"


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    with open("omegamind web.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/token-journey.html", response_class=HTMLResponse)
async def serve_token_journey():
    with open("token-journey.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/console", response_class=HTMLResponse)
async def serve_console():
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/api/metrics")
async def metrics():
    data = get_system_metrics()
    data["status"] = "ok" if (proc and proc.poll() is None) else "down"
    return data


@app.post("/api/reset")
async def reset_session(req: ResetRequest):
    with proc_lock:
        if proc and proc.poll() is None:
            proc.stdin.write(f"RESET|{req.session_id}\n")
            proc.stdin.flush()
            resp = proc.stdout.readline().strip()
            return {"status": resp}
    return JSONResponse({"error": "Engine not running"}, status_code=500)


@app.post("/api/chat")
async def chat_stream(req: ChatRequest):
    prompt_str = (
        f"{SYSTEM_PROMPT}User: {req.message.strip()}\nAssistant:"
        if req.is_first_turn
        else f"\nUser: {req.message.strip()}\nAssistant:"
    )
    token_ids = enc.encode_ordinary(prompt_str)
    tok_csv = ",".join(map(str, token_ids))
    stop_csv = f"{EOT}"

    cmd = (
        f"REQUEST|{req.session_id}|{tok_csv}|{req.max_tokens}|"
        f"{req.temperature:.3f}|{req.top_k}|{stop_csv}|"
        f"{req.top_p:.3f}|{req.repetition_penalty:.3f}\n"
    )

    def event_generator():
        global proc
        with proc_lock:
            if proc is None or proc.poll() is not None:
                start_engine()

            psutil.cpu_percent(interval=None)

            proc.stdin.write(cmd)
            proc.stdin.flush()

            byte_buffer = bytearray()
            full_text = ""
            ttft_ms = None
            gen_count = 0

            while True:
                line = proc.stdout.readline()
                if not line:
                    break
                line = line.strip()

                if line.startswith("ERROR"):
                    yield f"data: {json.dumps({'error': line})}\n\n"
                    break

                if line.startswith("TOKEN"):
                    parts = line.split()
                    tok_id = int(parts[1])
                    elapsed_ms = float(parts[2])
                    kv_pos = int(parts[3]) if len(parts) > 3 else 0

                    if tok_id == EOT:
                        continue

                    if ttft_ms is None:
                        ttft_ms = elapsed_ms

                    gen_count += 1
                    raw_b = vocab_bytes.get(tok_id, b"")
                    byte_buffer.extend(raw_b)

                    try:
                        chunk_str = byte_buffer.decode("utf-8")
                        byte_buffer.clear()
                    except UnicodeDecodeError:
                        continue

                    full_text += chunk_str

                    total_ms = round(elapsed_ms, 1)
                    tok_per_sec = round(gen_count / max(elapsed_ms / 1000.0, 1e-4), 1)
                    ms_per_tok = round(elapsed_ms / max(gen_count, 1), 1)
                    sys_m = get_system_metrics()

                    if "\nUser:" in full_text or "\nSystem:" in full_text:
                        while True:
                            rem = proc.stdout.readline()
                            if not rem or rem.startswith("DONE"):
                                break
                        yield f"data: {json.dumps({'done': True, 'tokens': gen_count, 'ttft_ms': round(ttft_ms, 1), 'total_ms': total_ms, 'tok_s': tok_per_sec, 'ms_per_tok': ms_per_tok, 'kv_pos': kv_pos, **sys_m})}\n\n"
                        break

                    payload = {
                        "token": chunk_str,
                        "ttft_ms": round(ttft_ms, 1),
                        "total_ms": total_ms,
                        "tok_s": tok_per_sec,
                        "ms_per_tok": ms_per_tok,
                        "kv_pos": kv_pos,
                        "tokens": gen_count,
                        **sys_m
                    }
                    yield f"data: {json.dumps(payload)}\n\n"

                elif line.startswith("DONE"):
                    parts = line.split()
                    total_tok = max(int(parts[1]), gen_count)
                    total_ms = round(float(parts[2]), 1)
                    kv_pos = int(parts[3]) if len(parts) > 3 else 0
                    tok_s = round(total_tok / max(total_ms / 1000.0, 1e-4), 1)
                    ms_per_tok = round(total_ms / max(total_tok, 1), 1)
                    sys_m = get_system_metrics()

                    yield f"data: {json.dumps({'done': True, 'tokens': total_tok, 'ttft_ms': round(ttft_ms or total_ms, 1), 'total_ms': total_ms, 'tok_s': tok_s, 'ms_per_tok': ms_per_tok, 'kv_pos': kv_pos, **sys_m})}\n\n"
                    break

    return StreamingResponse(event_generator(), media_type="text/event-stream")