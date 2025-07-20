import os
import time
import logging
import glob
import whisper
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from tqdm import tqdm
import torch

# 1. Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s', datefmt='%H:%M:%S')

# 2. Configuration
VIDEO_FOLDER = './'
WHISPER_MODEL_NAME = "base.en"
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
TOP_K_RESULTS = 3

# Ensure CUDA GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logging.info(f"🚀 Using device: {device}")
if device == 'cuda':
    logging.info(f"CUDA device: {torch.cuda.get_device_name(0)}")

# 3. Core Functions
def transcribe_videos(video_paths):
    logging.info(f"Loading Whisper model '{WHISPER_MODEL_NAME}' on {device}...")
    model = whisper.load_model(WHISPER_MODEL_NAME, device=device)
    logging.info("✅ Whisper model loaded successfully.")

    all_segments = []
    for video_path in tqdm(video_paths, desc="➡️ Transcribing Videos", unit="video"):
        tqdm.write(f"Processing: {os.path.basename(video_path)}")
        try:
            result = model.transcribe(video_path, fp16=(device == 'cuda'), verbose=False)
            for segment in result['segments']:
                all_segments.append({
                    'video_file': os.path.basename(video_path),
                    'start': segment['start'],
                    'text': segment['text']
                })
        except Exception as e:
            tqdm.write(f"❌ ERROR: Could not transcribe {os.path.basename(video_path)}. Reason: {e}")

    return pd.DataFrame(all_segments)

def create_cpu_index(df, model):
    logging.info("🧠 Generating embeddings on GPU...")
    embeddings = model.encode(df['text'].tolist(), show_progress_bar=True, device=device)
    embeddings_cpu = np.array(embeddings).astype('float32')
    logging.info(f"✅ Embeddings moved to CPU. Shape: {embeddings_cpu.shape}")

    d = embeddings_cpu.shape[1]
    index = faiss.IndexFlatL2(d)
    logging.info("🛠️ Building FAISS CPU index...")
    index.add(embeddings_cpu)
    logging.info(f"✅ FAISS CPU index built with {index.ntotal} vectors.")
    return index

def search_cpu(query, index, model, df, top_k):
    logging.info(f"🔍 Searching for query: '{query}'")
    start_time = time.time()

    query_embedding = model.encode([query], device=device, show_progress_bar=False)
    query_cpu = np.array(query_embedding).astype('float32')

    distances, indices = index.search(query_cpu, top_k)
    logging.info(f"✅ CPU Search completed in {(time.time() - start_time) * 1000:.2f} ms.")

    results = [df.iloc[idx] for idx in indices[0] if idx != -1]
    return pd.DataFrame(results)

# 4. Main
if __name__ == "__main__":
    logging.info("🚀 Starting Interactive Video Search Engine...")
    video_paths = glob.glob(os.path.join(VIDEO_FOLDER, '*.mp4'))
    if not video_paths:
        logging.error(f"❌ No .mp4 videos found in {os.getcwd()}. Exiting.")
        exit(1)
    logging.info(f"Found {len(video_paths)} videos to process.")

    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
    transcription_df = transcribe_videos(video_paths)

    if transcription_df.empty:
        logging.error("❌ No text was transcribed from the videos. Exiting.")
        exit(1)

    cpu_index = create_cpu_index(transcription_df, embedding_model)
    logging.info("✅ All videos have been indexed. Ready for search.")

    while True:
        print("\n")
        user_query = input("Enter your search query (or type 'exit' to quit): ")
        if user_query.lower() in ['exit', 'quit']:
            logging.info("Exiting search loop.")
            break

        search_results_df = search_cpu(user_query, cpu_index, embedding_model, transcription_df, TOP_K_RESULTS)
        print("\n" + "="*50 + "\n✨ Top Search Results ✨\n" + "="*50 + "\n")
        if not search_results_df.empty:
            for idx, row in search_results_df.iterrows():
                ts = time.strftime('%H:%M:%S', time.gmtime(row['start']))
                print(f"🏆 Result #{idx+1}\n🎬 Video: {row['video_file']}\n🕒 Timestamp: {ts}\n📝 \"{row['text'].strip()}\"\n" + "-"*50)
        else:
            print("No relevant segments found. Try another query.")
    logging.info("✅ Script finished.")
