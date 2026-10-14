"""
Script rápido para testar a busca semântica de versículos diretamente pelo terminal.
Não utiliza Whisper, não consome microfone e não sobrecarrega a CPU.

Uso:
    python test_search.py
    ou
    python test_search.py "O Senhor é o meu pastor"
"""

import sys
import json
import time
from pathlib import Path

# Garante compatibilidade com UTF-8 no terminal Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

CACHE_DIR = Path("bible_cache")

def main():
    print("=" * 60)
    print("  BibleRadar - Teste Rapido de Pesquisa Semantica (CLI)")
    print("  (Sem audio / Sem Whisper / Leve para qualquer CPU)")
    print("=" * 60)

    idx_path = CACHE_DIR / "nvi_faiss.index"
    vrs_path = CACHE_DIR / "nvi_verses.json"

    if not idx_path.exists() or not vrs_path.exists():
        print("[Aviso] Cache da versao NVI nao encontrado em 'bible_cache'.")
        print("Inicie o server.py uma vez para descarregar e indexar a Biblia.")
        return

    print("A carregar modelo de embeddings...")
    t0 = time.time()
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    index = faiss.read_index(str(idx_path))
    verses = json.loads(vrs_path.read_text(encoding="utf-8"))
    print(f"[OK] Pronto em {round(time.time() - t0, 1)}s! {index.ntotal} versiculos indexados.\n")

    def run_query(query: str, threshold: float = 0.65, top_k: int = 5):
        t_start = time.time()
        emb = model.encode([query.strip()], normalize_embeddings=True, convert_to_numpy=True)
        scores, ids = index.search(emb, top_k)
        elapsed = round((time.time() - t_start) * 1000, 1)

        print(f"\n--- Busca: \"{query}\" ({elapsed} ms) ---")
        found = 0
        for score, idx in zip(scores[0], ids[0]):
            if score >= threshold and idx >= 0:
                v = verses[idx]
                pct = int(score * 100)
                print(f"[{pct}%] {v['ref']}")
                print(f"   \"{v['text']}\"\n")
                found += 1
        if found == 0:
            print("  Nenhum versiculo atingiu o limiar minimo de similaridade.\n")

    # Se passou texto via linha de comando
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        run_query(query)
        return

    # Modo interativo
    print("Digite frases ou palavras que deseja testar (ou 'sair' para encerrar):")
    print("-" * 60)
    while True:
        try:
            q = input("\nTexto > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nA encerrar...")
            break

        if not q:
            continue
        if q.lower() in ("sair", "exit", "quit"):
            break

        run_query(q)

if __name__ == "__main__":
    main()
