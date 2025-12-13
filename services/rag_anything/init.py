import asyncio
import json
import os
from raganything import RAGAnything, RAGAnythingConfig
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc
from lightrag import LightRAG
from lightrag.kg.shared_storage import initialize_pipeline_status
from raganything.modalprocessors import ImageModalProcessor, TableModalProcessor

def get_api_key():
    if os.path.exists(SECRETS_FILE):
        with open(SECRETS_FILE, "r") as f:
            secrets = json.load(f)
            os.environ["OPENAI_API_KEY"] = secrets.get("OPENAI_API_KEY", "")
            return  secrets.get("OPENAI_API_KEY", None)

LLM_MODEL = os.getenv("OPENAI_MODEL", "llama3.1:8b")
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "bge-m3")
RAG_KB_DIR = os.getenv("RAG_KB_DIR", "/app/kb")
MANIFEST_FILE = "/app/data/kb-manifest.json"
WORKING_DIR = os.getenv("RAG_WORKING_DIR", "/app/data/rag_storage")
OUTPUT_DIR = os.getenv("RAG_OUTPUT_DIR", "/app/data/output")
SECRETS_FILE = os.getenv("RAG_SECRETS_FILE", "/app/data/rag_secrets.json")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://cis-ollama:11434/v1")


            # Add other secrets as needed


async def init_rag(load_lightrag: bool=False):    
    OPENAI_KEY = get_api_key()
    light_rag = None
    rag = None
    # Configure RAGAnything with OpenAI embedding and completion functions
    config = RAGAnythingConfig(
        working_dir=WORKING_DIR,
        parser="mineru",  # Parser selection: mineru or docling
        parse_method="auto",  # Parse method: auto, ocr, or txt
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
    )

    #define LLM model function
    def llm_model_func(prompt: str, system_prompt: str=None, history_messages=[], **kwargs) -> str:
        return openai_complete_if_cache(
            LLM_MODEL,
            prompt,
            api_key=OPENAI_KEY,
            system_prompt=system_prompt,
            history_messages=history_messages,
            base_url=OPENAI_BASE_URL,
            **kwargs,
        )
    
    def vision_model_func(prompt: str, system_prompt: str=None, history_messages=[], image_data=None, messages=None, **kwargs) -> str:
        if messages:
            return openai_complete_if_cache(
                LLM_MODEL,
                "",
                system_prompt=system_prompt,
                history_messages=[],
                messages=messages,
                api_key=OPENAI_KEY,
                base_url=OPENAI_BASE_URL,
                **kwargs,
            )
        elif image_data:
            return openai_complete_if_cache(
                LLM_MODEL,
                "",
                system_prompt=None,
                history_messages=[],
                messages=[{"role": "system", "content": system_prompt}
                          if system_prompt else None,
                          {
                              "role": "user",
                              "content": [
                                  {
                                      "type": "text",
                                      "text": prompt
                                  },
                                  {
                                      "type": "image_url",
                                      "image_url": {
                                          "url": f"data:image/jpeg;base64,{image_data}"
                                      },                                      
                                  },
                              ],
                          } if image_data
                          else {"role": "user", "content": prompt},
                          ],
                api_key=OPENAI_KEY,
                base_url=OPENAI_BASE_URL,
                **kwargs,
            )
        else:
            return llm_model_func(prompt, system_prompt, history_messages, **kwargs)

    embedding_func = EmbeddingFunc(
        embedding_dim=1024,
        max_token_size=8192,
        func=lambda texts: openai_embed(
            texts,
            model=EMBEDDING_MODEL,
            api_key=OPENAI_KEY,
            base_url=OPENAI_BASE_URL,
        ),
    )
    if load_lightrag:
        light_rag = LightRAG(
            working_dir=WORKING_DIR,
            llm_model_func=llm_model_func,
            embedding_func=embedding_func,
        )
        await light_rag.initialize_storages()
        await initialize_pipeline_status()
        print("Initialized LightRAG...")

        rag = RAGAnything(
            lightrag=light_rag,
            vision_model_func=vision_model_func,
        )
    else:
        rag = RAGAnything(
            config=config,
            llm_model_func=llm_model_func,
            embedding_func=embedding_func,
            vision_model_func=vision_model_func,     
        )
    return rag

def get_manifest():
    if not os.path.exists(MANIFEST_FILE):        
        return {}
    with open(MANIFEST_FILE, "r") as f:
        return json.load(f)

def save_manifest(manifest):
    with open(MANIFEST_FILE, "w") as f:
        json.dump(manifest, f, indent=4)

async def main():
    
    rag = await init_rag()
    # Load document manifest to see what documents are already indexed
    loaded_docs = get_manifest()
    print(f"Loaded {len(loaded_docs)} documents from manifest.")
    # Index documents if not already indexed
    new_files = []
    for root, dirs, files in os.walk(RAG_KB_DIR):
        for file in files:
            file_path = os.path.join(root, file)
            last_modified = os.path.getmtime(file_path)
            if file_path not in loaded_docs or loaded_docs[file_path] != last_modified:
                new_files.append(file_path)
    if new_files:
        print(f"Indexing {len(new_files)} new/updated documents...")
        for file_path in new_files:
            try:
                await rag.process_document_complete(file_path, output_dir=OUTPUT_DIR, parse_method="auto")
                loaded_docs[file_path] = os.path.getmtime(file_path)
                print(f"Indexed: {file_path}")
            except Exception as e:
                print(f"Failed to index {file_path}: {e}")

    # save loaded docs to manifest
    save_manifest(loaded_docs)

if __name__ == "__main__":
    asyncio.run(main())