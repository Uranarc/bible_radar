"""
BibleRadar - Servidor de transcrição em tempo real + busca de versículos
Requisitos:
    pip install fastapi uvicorn websockets sounddevice numpy openai-whisper
    pip install sentence-transformers faiss-cpu torch requests

Bíblias: descarregadas automaticamente na primeira utilização de cada versão.
"""

import asyncio
import json
import threading
import queue
import time
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd
import whisper
import faiss
import requests
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
import uvicorn

# ---------------------------------------------------------------------------
# CONFIGURAÇÕES GERAIS
# ---------------------------------------------------------------------------
SAMPLE_RATE          = 16000   # Hz exigido pelo Whisper
CHUNK_SECONDS        = 6       # segundos de áudio por chunk
OVERLAP_SECONDS      = 2       # sobreposição entre chunks
TOP_K                = 7       # versículos no ranking
SIMILARITY_THRESHOLD = 0.75    # score mínimo (0-1) - padrão 75%
CONTEXT_WINDOW       = 600     # caracteres para contexto de busca expandido
WHISPER_MODEL        = "small" # tiny | base | small | medium | large

CACHE_DIR = Path("bible_cache")
CACHE_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# CATÁLOGO DE VERSÕES DISPONÍVEIS
# ---------------------------------------------------------------------------
BIBLE_VERSIONS = {
    "nvi": {
        "label": "NVI — Nova Versão Internacional",
        "url": "https://raw.githubusercontent.com/thiagobodruk/biblia/master/json/nvi.json",
    },
    "acf": {
        "label": "ACF — Almeida Corrigida Fiel",
        "url": "https://raw.githubusercontent.com/thiagobodruk/biblia/master/json/acf.json",
    },
    "aa": {
        "label": "AA — Almeida Atualizada",
        "url": "https://raw.githubusercontent.com/thiagobodruk/biblia/refs/heads/master/json/aa.json",
    },
    "naa": {
        "label": "NAA — Nova Almeida Atualizada",
        "url": "https://raw.githubusercontent.com/thiagobodruk/biblia/master/json/naa.json",
    },
    "nvt":{
        "label": "NVT - Nova Versão Transformadora",
        "url": "https://raw.githubusercontent.com/damarals/biblias/refs/heads/main/inst/json/NVT.json"
    },
}

BOOK_NAMES_PT = {
    "Genesis": "Gênesis", "Exodus": "Êxodo", "Leviticus": "Levítico",
    "Numbers": "Números", "Deuteronomy": "Deuteronômio", "Joshua": "Josué",
    "Judges": "Juízes", "Ruth": "Rute", "1 Samuel": "1 Samuel",
    "2 Samuel": "2 Samuel", "1 Kings": "1 Reis", "2 Kings": "2 Reis",
    "1 Chronicles": "1 Crônicas", "2 Chronicles": "2 Crônicas",
    "Ezra": "Esdras", "Nehemiah": "Neemias", "Esther": "Ester",
    "Job": "Jó", "Psalms": "Salmos", "Proverbs": "Provérbios",
    "Ecclesiastes": "Eclesiastes", "Song of Solomon": "Cantares",
    "Isaiah": "Isaías", "Jeremiah": "Jeremias", "Lamentations": "Lamentações",
    "Ezekiel": "Ezequiel", "Daniel": "Daniel", "Hosea": "Oséias",
    "Joel": "Joel", "Amos": "Amós", "Obadiah": "Obadias",
    "Jonah": "Jonas", "Micah": "Miquéias", "Nahum": "Naum",
    "Habakkuk": "Habacuque", "Zephaniah": "Sofonias", "Haggai": "Ageu",
    "Zechariah": "Zacarias", "Malachi": "Malaquias", "Matthew": "Mateus",
    "Mark": "Marcos", "Luke": "Lucas", "John": "João", "Acts": "Atos",
    "Romans": "Romanos", "1 Corinthians": "1 Coríntios",
    "2 Corinthians": "2 Coríntios", "Galatians": "Gálatas",
    "Ephesians": "Efésios", "Philippians": "Filipenses",
    "Colossians": "Colossenses", "1 Thessalonians": "1 Tessalonicenses",
    "2 Thessalonians": "2 Tessalonicenses", "1 Timothy": "1 Timóteo",
    "2 Timothy": "2 Timóteo", "Titus": "Tito", "Philemon": "Filêmon",
    "Hebrews": "Hebreus", "James": "Tiago", "1 Peter": "1 Pedro",
    "2 Peter": "2 Pedro", "1 John": "1 João", "2 John": "2 João",
    "3 John": "3 João", "Jude": "Judas", "Revelation": "Apocalipse",
}

# ---------------------------------------------------------------------------
# CARREGAMENTO E INDEXAÇÃO
# ---------------------------------------------------------------------------

def _parse_raw_bible(raw: list, version_id: str) -> list[dict]:
    verses = []
    for book in raw:
        book_name    = book.get("name", book.get("abbrev", "?"))
        book_name_pt = BOOK_NAMES_PT.get(book_name, book_name)
        for chap_idx, chapter in enumerate(book.get("chapters", []), 1):
            for verse_idx, text in enumerate(chapter, 1):
                if isinstance(text, str) and text.strip():
                    verses.append({
                        "ref":     f"{book_name_pt} {chap_idx}:{verse_idx}",
                        "book":    book_name_pt,
                        "chapter": chap_idx,
                        "verse":   verse_idx,
                        "text":    text.strip(),
                        "version": version_id,
                    })
    return verses


def load_version_verses(version_id: str) -> list[dict]:
    cfg = BIBLE_VERSIONS[version_id]
    cache_path = CACHE_DIR / f"{version_id}_raw.json"

    if "file" in cfg:
        raw = json.loads(Path(cfg["file"]).read_text(encoding="utf-8"))
        return _parse_raw_bible(raw, version_id)

    if not cache_path.exists():
        print(f"[Bible:{version_id}] Descarregando de {cfg['url']} ...")
        r = requests.get(cfg["url"], timeout=30)
        r.raise_for_status()
        cache_path.write_bytes(r.content)

    raw = json.loads(cache_path.read_text(encoding="utf-8-sig"))
    return _parse_raw_bible(raw, version_id)


def build_or_load_version_index(version_id: str, verses: list[dict],
                                 model: SentenceTransformer):
    emb_path = CACHE_DIR / f"{version_id}_embeddings.npy"
    idx_path = CACHE_DIR / f"{version_id}_faiss.index"
    vrs_path = CACHE_DIR / f"{version_id}_verses.json"

    if emb_path.exists() and idx_path.exists() and vrs_path.exists():
        print(f"[Index:{version_id}] Cache encontrado, carregando...")
        index   = faiss.read_index(str(idx_path))
        ordered = json.loads(vrs_path.read_text(encoding="utf-8"))
        return index, ordered

    print(f"[Index:{version_id}] Gerando embeddings ({len(verses)} versículos)...")
    texts      = [v["text"] for v in verses]
    embeddings = model.encode(
        texts, batch_size=256, show_progress_bar=True,
        normalize_embeddings=True, convert_to_numpy=True,
    )
    np.save(str(emb_path), embeddings)

    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, str(idx_path))
    vrs_path.write_text(json.dumps(verses, ensure_ascii=False), encoding="utf-8")

    print(f"[Index:{version_id}] ✅ {index.ntotal} vetores indexados.")
    return index, verses

# ---------------------------------------------------------------------------
# ESTADO GLOBAL
# ---------------------------------------------------------------------------

class BibleRadar:
    def __init__(self):
        self.whisper_model = None
        self.embed_model: Optional[SentenceTransformer] = None

        # version_id -> {"index": faiss.Index, "verses": list[dict]}
        self.version_data: dict[str, dict] = {}
        self.active_versions: set[str]     = set()

        self.audio_queue        = queue.Queue()
        self.clients: list[WebSocket] = []
        self.is_listening       = False
        self.transcript_buffer  = ""
        self.device_index: Optional[int] = None
        self.ready              = False
        self.loading_status     = "A iniciar..."

        # Configurações dinâmicas
        self.similarity_threshold: float = SIMILARITY_THRESHOLD
        self.context_window: int         = CONTEXT_WINDOW

    # ── Carregamento ─────────────────────────────────────────────────────────

    def load_models(self):
        self.loading_status = "A carregar Whisper..."
        self.whisper_model  = whisper.load_model(WHISPER_MODEL)

        self.loading_status = "A carregar modelo de embeddings..."
        self.embed_model    = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        )

        # Carrega a versão default (primeira do catálogo)
        default = next(iter(BIBLE_VERSIONS))
        self._load_version_sync(default)
        self.active_versions.add(default)

        self.ready          = True
        self.loading_status = "Pronto"
        print("[BibleRadar] ✅ Pronto!")

    def _load_version_sync(self, version_id: str):
        """Carrega e indexa uma versão (bloqueante — usar em thread)."""
        if version_id in self.version_data:
            return
        cfg = BIBLE_VERSIONS.get(version_id)
        if not cfg:
            raise ValueError(f"Versão desconhecida: {version_id}")
        self.loading_status = f"A carregar {cfg['label']}..."
        verses = load_version_verses(version_id)
        index, verses = build_or_load_version_index(
            version_id, verses, self.embed_model
        )
        self.version_data[version_id] = {"index": index, "verses": verses}
        self.loading_status = "Pronto"
        print(f"[BibleRadar] Versão '{version_id}' disponível.")

    def enable_version(self, version_id: str):
        if version_id not in BIBLE_VERSIONS:
            return False
        if version_id not in self.version_data:
            threading.Thread(
                target=self._load_version_sync, args=(version_id,), daemon=True
            ).start()
        self.active_versions.add(version_id)
        return True

    def disable_version(self, version_id: str):
        self.active_versions.discard(version_id)

    # ── Busca ────────────────────────────────────────────────────────────────

    def search(self, text: str, threshold: Optional[float] = None, top_k: int = TOP_K) -> list[dict]:
        min_thresh = threshold if threshold is not None else self.similarity_threshold
        if not text.strip() or not self.active_versions or not self.embed_model:
            return []

        emb = self.embed_model.encode(
            [text.strip()], normalize_embeddings=True, convert_to_numpy=True
        )
        all_results = []
        for vid in list(self.active_versions):
            vdata = self.version_data.get(vid)
            if not vdata:
                continue
            scores, idxs = vdata["index"].search(emb, top_k)
            for score, idx in zip(scores[0], idxs[0]):
                if score >= min_thresh and idx >= 0:
                    v = vdata["verses"][idx]
                    all_results.append({**v, "score": float(score)})

        all_results.sort(key=lambda x: x["score"], reverse=True)
        return all_results[:top_k]

    # ── WebSocket broadcast ──────────────────────────────────────────────────

    async def broadcast(self, msg: dict):
        data = json.dumps(msg, ensure_ascii=False)
        dead = []
        for ws in self.clients:
            try:
                await ws.send_text(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.clients.remove(ws)


radar = BibleRadar()
app   = FastAPI()
main_loop = None

# ---------------------------------------------------------------------------
# ÁUDIO + TRANSCRIÇÃO
# ---------------------------------------------------------------------------

def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"[Audio] {status}")
    radar.audio_queue.put(indata.copy())


def transcription_loop():
    chunk_samples = int(SAMPLE_RATE * CHUNK_SECONDS)
    overlap_samples = int(SAMPLE_RATE * OVERLAP_SECONDS)

    buffer = np.zeros(0, dtype=np.float32)

    print("[Transcricao] Thread iniciada.")

    while radar.is_listening:
        try:
            data = radar.audio_queue.get(timeout=0.5)
            buffer = np.concatenate([buffer, data.flatten()])
        except queue.Empty:
            continue

        if len(buffer) < chunk_samples:
            continue

        chunk = buffer[:chunk_samples].copy()
        buffer = buffer[chunk_samples - overlap_samples:]

        peak = float(np.max(np.abs(chunk)))
        rms = float(np.sqrt(np.mean(chunk ** 2)))

        if peak < 0.001:
            print("[Audio] Chunk sem sinal significativo.")
            continue

        try:
            result = radar.whisper_model.transcribe(
                chunk,
                language="pt",
                fp16=False,
                condition_on_previous_text=False,
            )
            text = result.get("text", "").strip()
            print(f"[Whisper] {text!r}")
        except Exception as exc:
            import traceback
            print(f"[Whisper] Erro: {exc}")
            traceback.print_exc()
            continue

        if not text:
            continue

        radar.transcript_buffer = (
            radar.transcript_buffer + " " + text
        ).strip()

        # Multi-window search:
        # 1. Janela curta (últimos ~180 caracteres) para citações diretas
        # 2. Janela ampla (últimos context_window caracteres) para passagens longas e contexto
        short_len = 180
        text_short = radar.transcript_buffer[-short_len:].strip()
        text_long  = radar.transcript_buffer[-radar.context_window:].strip()

        combined_matches: dict[tuple, dict] = {}

        matches_short = radar.search(text_short)
        for m in matches_short:
            key = (m["version"], m["ref"])
            combined_matches[key] = m

        if text_long != text_short:
            matches_long = radar.search(text_long)
            for m in matches_long:
                key = (m["version"], m["ref"])
                if key not in combined_matches or m["score"] > combined_matches[key]["score"]:
                    combined_matches[key] = m

        matches = sorted(
            combined_matches.values(),
            key=lambda x: x["score"],
            reverse=True
        )[:TOP_K]

        if main_loop is not None:
            future = asyncio.run_coroutine_threadsafe(
                radar.broadcast({
                    "type": "update",
                    "transcript": text,
                    "fullBuffer": radar.transcript_buffer[-1000:],
                    "matches": matches,
                    "timestamp": time.time(),
                }),
                main_loop,
            )

            try:
                future.result(timeout=10)
            except Exception as exc:
                print(f"[WebSocket] Erro no envio: {exc}")

    print("[Transcricao] Thread terminada.")

# ---------------------------------------------------------------------------
# ENDPOINTS
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup():
    global main_loop
    main_loop = asyncio.get_running_loop()

    threading.Thread(
        target=radar.load_models,
        daemon=True,
    ).start()


@app.get("/api/status")
async def api_status():
    devices = []
    try:
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0:
                devices.append({"id": i, "name": d["name"]})
    except Exception:
        pass

    versions_info = [
        {
            "id":     vid,
            "label":  cfg["label"],
            "loaded": vid in radar.version_data,
            "active": vid in radar.active_versions,
        }
        for vid, cfg in BIBLE_VERSIONS.items()
    ]

    return {
        "ready":               radar.ready,
        "loadingStatus":       radar.loading_status,
        "listening":           radar.is_listening,
        "devices":             devices,
        "deviceIndex":         radar.device_index,
        "versions":            versions_info,
        "activeVersions":      list(radar.active_versions),
        "similarityThreshold": radar.similarity_threshold,
        "contextWindow":       radar.context_window,
    }


@app.post("/api/settings")
async def update_settings(body: dict):
    if "similarityThreshold" in body:
        thresh = float(body["similarityThreshold"])
        radar.similarity_threshold = max(0.1, min(1.0, thresh))
    if "contextWindow" in body:
        cw = int(body["contextWindow"])
        radar.context_window = max(100, min(3000, cw))

    return {
        "ok": True,
        "similarityThreshold": radar.similarity_threshold,
        "contextWindow": radar.context_window,
    }


@app.post("/api/search")
async def manual_search(body: dict):
    text = body.get("text", "").strip()
    threshold = body.get("threshold", None)
    if threshold is not None:
        try:
            threshold = float(threshold)
        except (ValueError, TypeError):
            threshold = None

    if not text:
        return {"ok": False, "error": "Texto vazio.", "results": []}

    matches = radar.search(text, threshold=threshold)
    return {
        "ok": True,
        "query": text,
        "results": matches,
    }


@app.post("/api/versions/{version_id}/enable")
async def enable_version(version_id: str):
    ok = radar.enable_version(version_id)
    return {"ok": ok, "active": list(radar.active_versions)}


@app.post("/api/versions/{version_id}/disable")
async def disable_version(version_id: str):
    radar.disable_version(version_id)
    return {"ok": True, "active": list(radar.active_versions)}


@app.post("/api/start")
async def start_listening(body: dict = {}):
    if not radar.ready:
        return {
            "ok": False,
            "error": f"Modelos ainda a carregar: {radar.loading_status}",
        }

    if radar.is_listening:
        return {
            "ok": False,
            "error": "Já está a ouvir.",
        }

    device_id = body.get("deviceId", None)

    try:
        if device_id is not None:
            device_id = int(device_id)
            info = sd.query_devices(device_id)

            if info["max_input_channels"] < 1:
                return {
                    "ok": False,
                    "error": "O dispositivo selecionado não tem entrada de áudio.",
                }

        sd.check_input_settings(
            device=device_id,
            channels=1,
            dtype="float32",
            samplerate=SAMPLE_RATE,
        )

    except Exception as exc:
        return {
            "ok": False,
            "error": f"Configuração de áudio inválida: {exc}",
        }

    while True:
        try:
            radar.audio_queue.get_nowait()
        except queue.Empty:
            break

    radar.device_index = device_id
    radar.transcript_buffer = ""
    radar.is_listening = True

    def run():
        try:
            print(
                f"[Audio] A abrir dispositivo {device_id} "
                f"a {SAMPLE_RATE} Hz."
            )

            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                device=device_id,
                blocksize=int(SAMPLE_RATE * 0.5),
                callback=audio_callback,
            ):
                print("[Audio] Captura iniciada.")
                transcription_loop()

        except Exception as exc:
            import traceback
            print(f"[Audio] Erro fatal: {exc}")
            traceback.print_exc()

        finally:
            radar.is_listening = False
            print("[Audio] Captura terminada.")

    threading.Thread(
        target=run,
        daemon=True,
    ).start()

    return {
        "ok": True,
        "deviceId": device_id,
        "message": "Pedido de captura aceite; a confirmar nos logs.",
    }


@app.post("/api/stop")
async def stop_listening():
    radar.is_listening = False
    radar.transcript_buffer = ""
    await radar.broadcast({"type": "stopped"})
    return {"ok": True}


@app.post("/api/clear")
async def clear_buffer():
    radar.transcript_buffer = ""
    await radar.broadcast({"type": "cleared"})
    return {"ok": True}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    radar.clients.append(ws)
    await ws.send_text(json.dumps({
        "type":      "connected",
        "ready":     radar.ready,
        "listening": radar.is_listening,
    }))
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        if ws in radar.clients:
            radar.clients.remove(ws)


@app.get("/")
async def index():
    return FileResponse("index.html")


if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  BibleRadar pronto!")
    print("  Acesse no browser: http://127.0.0.1:7890")
    print("=" * 55 + "\n")
    uvicorn.run(app, host="127.0.0.1", port=7890, log_level="info")
