# pip install unsloth
# pip install chromadb
# pip install langchain_text_splitters
# pip install sentence_transformers

from unsloth import FastModel
import os
import glob
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict, Any
import argparse
from transformers import TextStreamer
import re
import time
import csv
import textstat
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

BASEDIR_PATH = os.path.dirname(__file__)
QAPair_PATH = os.path.join(BASEDIR_PATH, "QAPair.json")
PROMT_PATH = os.path.join(BASEDIR_PATH, "promt.txt")
QUESTIONS_PATH = os.path.join(BASEDIR_PATH, "questions.txt")

APPLICATION_LIST_PATH = os.path.join(BASEDIR_PATH, "application_list.txt")

CHROMADB_PATH = "chroma_store"
CHROMA_DIR = os.path.join(BASEDIR_PATH, CHROMADB_PATH)
COLLECTION_NAME = "files_chunks"
#DATASTORE_PATH = "./storage"
DATASTORE_PATH = os.path.join(BASEDIR_PATH, "storage_test") # TODO edit

with open(QAPair_PATH, "r", encoding="utf-8") as jf:
    data = json.load(jf)

QAPair = {
    item["folder_id"]: item["questions"]
    for item in data["application_summaries"]
}

vectorizer = TfidfVectorizer()

# METRICS

RUN_NUM = 1
MODEL_NAME = "phi"

RANK_B = 0
RANK_E = 0

REQUEST_B = 0
REQUEST_E = 0

FULL_PROMT_LEN_C = 0
CONTEXT_LEN_C = 0 #

CONTEXT_CONTENT = 0

INPUT_TOKEN_COUNT = 0
OUTPUT_TOKEN_COUNT = 0

#print(CHROMA_DIR)
#print(DATASTORE_PATH)

# if N_CHUNKS = -1 will use recomended number of cunks for short context window, else will use your number
N_CHUNKS = -1

EMBEDING_MODEL = "all-mpnet-base-v2"
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 300

# value from 0.0 to 1.0. The higher the value, the more the model hallucinates
MODEL_TEMPERATURE = 0.8

def add_record_to_csv(filename, **fields):
    # Проверяем, нужно ли записать заголовки (если файл ещё не создан)
    write_header = False
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            pass
    except FileNotFoundError:
        write_header = True

    # Добавляем запись
    with open(filename, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields.keys())

        if write_header:
            writer.writeheader()

        writer.writerow(fields)

def get_vector_storage():
    
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBEDING_MODEL
    )
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_func,
    )
    return collection


def add_chunks_to_db(collection, documents, metadatas, ids):
    if documents:
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
        )
        print(f"Chunks added: {len(documents)}")
    else:
        print("No files .md found")


def create_chunks(markdown_dir, collection, global_chunk_index = 0):
    
    md_files = glob.glob(os.path.join(markdown_dir, "**", "*.md"), recursive=True)
    
    # Chunk splitter setup. It sepatates files based on articles. 
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    
    for file_path in md_files:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    
        if not text.strip():
            continue
        
        documents, metadatas, ids = [], [], []
        chunks = splitter.split_text(text)
        
        application_name = file_path.split("/")[-2]
        
        for local_chunk_index, chunk in enumerate(chunks):
            doc_id = f"md_{global_chunk_index}"
            
            documents.append(chunk)
            metadatas.append({
                "path": file_path,
                "filename": os.path.basename(file_path),
                "chunk": local_chunk_index,   # chunk number inside file
                "application": application_name,

            })
            
            ids.append(doc_id)
            global_chunk_index += 1
            
        add_chunks_to_db(collection, documents, metadatas, ids)
        
def populate_vector_storage():
    
    collection = get_vector_storage()
    print("DEBUG: db created")
    
    create_chunks(DATASTORE_PATH, collection, 0)
    print("DEBUG: chunks created and added")

def init_model(model_alias="gemma"):
    
    model_url = ""
    max_seq_len = 2048
    max_chunks_for_short_context_win = 1
    
    if(model_alias == "gemma"):
        model_url = "nhannguyen2730/gemma3-4b-qlora"
        max_seq_len = 8000
        max_chunks_for_short_context_win = 10
    elif(model_alias == "phi"):
        model_url = "nhannguyen2730/phi-3-mini-instruct-qlora-tc"
        max_seq_len = 4000
        max_chunks_for_short_context_win = 6
    else:
        print("Unknown model alias")
        return None, None, 0
        
    model, tokenizer = FastModel.from_pretrained(
    model_name = model_url,
    load_in_4bit = True,
    )
    #   max_seq_length = max_seq_len,
    
    return model, tokenizer, max_chunks_for_short_context_win

RANK_MODEL, RANK_TOKENIZER, _ = init_model("gemma")

def rank_chunks(query, documents):
    result = {}
    if True:
        for i, doc_id in enumerate(documents["ids"]):
            chunks_text = documents["documents"][i]
            promt = f"""
            Query:
            \"\"\"{query}\"\"\"
            
            Document:
            \"\"\"{chunks_text}\"\"\"
            
            Evaluate how useful this document is for answering the query.
            
            Give ONE number from 0 to 9, where:
            0 — not relevant at all,
            9 — extremely relevant and useful.
            
            Reply with the number only.
            """
            messages = [    
            {
                "role": "system",
                "content": [{"type": "text", "text": " You are helping to rank documents by how relevant they are to a user query."}]
            },
            {
            "role": "user",
            "content": [{"type" : "text", "text" : promt,}]
            }]
            
            inputs = RANK_TOKENIZER.apply_chat_template(
                messages,
                add_generation_prompt = True, # Must add for generation
                return_tensors = "pt",
                tokenize = True,
                return_dict = True,
            ).to("cuda")
        
            outputs = RANK_MODEL.generate(
                **inputs,
                max_new_tokens = 128, # Increase for longer outputs!
                temperature = MODEL_TEMPERATURE, top_p = 0.9, top_k = 40,
            )
        
            # decoded answer of the model
            generated_text = RANK_TOKENIZER.decode(
                outputs[0][inputs["input_ids"].shape[-1]:],
                skip_special_tokens=True,
            )
            m = re.search(r"([0-9])", generated_text)
            score = int(m.group(1)) if m else 0
            
            #print(f"RANK {doc_id} = {score}")
            result[doc_id] = score
    return result

def summarize_chunks(query, document):
    
    result_summary = ""
    if True:
        promt = f"""
        Document:
        \"\"\"{document}\"\"\"
        
        Focus on the facts that may help answer the query:
        \"\"\"{query}\"\"\"
        
        Output 3–5 concise bullet points.
        """
        messages = [    
        {
            "role": "system",
            "content": [{"type": "text", "text": "Create a short, factual summary of the text fragment."}]
        },
        {
            "role": "user",
            "content": [{"type" : "text", "text" : promt,}]
        }]
        inputs = RANK_TOKENIZER.apply_chat_template(
            messages,
            add_generation_prompt = True, # Must add for generation
            return_tensors = "pt",
            tokenize = True,
            return_dict = True,
        ).to("cuda")

        outputs = RANK_MODEL.generate(
            **inputs,
            max_new_tokens = 64, # Increase for longer outputs!
            temperature = MODEL_TEMPERATURE, top_p = 0.9, top_k = 40,
        )
        generated_text = RANK_TOKENIZER.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        )

        result_summary += "\n" + generated_text
        #print(f"RESULT SUMMARY {result_summary}")
    return result_summary
        
    

def find_chunks(query, collection, n_results, application_id=None):

    result = None
    if application_id:
        #print(f"DEBUG: chunks for: {application_id} extracted")
        result = collection.query(
            query_texts=[query],
            n_results=20,
            where={"application": application_id},
            include=["documents", "metadatas", "distances"],
        )
    else:
        result = collection.query(
            query_texts=[query],
            n_results=20,
            include=["documents", "metadatas", "distances"],
        )

    chunks = {
        "ids": result["ids"][0],
        "documents": result["documents"][0],
        "metadatas": result["metadatas"][0],
        "distances": result["distances"][0],
    }
    ranked = rank_chunks(query, chunks)
    top_n = dict(sorted(ranked.items(), key=lambda x: x[1], reverse=True)[:n_results])

    #print(f"top_{n_results}: {top_n}")
    top_ids = list(top_n.keys())
    #print(top_ids)
    
    top_result = collection.get(
    ids=top_ids,
    include=["documents", "metadatas"],
    )

    #print(f"From top_result {top_result}")
    
    top_chunks = {
        "ids": top_result["ids"],
        "documents": top_result["documents"],
        "metadatas": top_result["metadatas"],
    }

    #print(f"BEFORE RANKING: {chunks["ids"]}, AFTER RANKING {top_result["ids"]}")
    return top_chunks
    
def prepare_promt(query, n_chunks, question_n, application_id=None):
    
    # get ChromdDB object
    collection = get_vector_storage()
    
    promt = ""
    with open(PROMT_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()
    promt = lines[question_n - 1].rstrip("\n")

    #print(f"PROMT FOR Q{question_n}: {promt}")
    
    # Searching for context based on query
    result = find_chunks(query, collection, n_chunks, application_id)    
    context = ""
    for i, doc_id in enumerate(result["ids"]):
        context += "\n\n" + result["documents"][i]
        #print(result["metadatas"][i]["application"]) #TODO debug

    global CONTEXT_LEN_C
    CONTEXT_LEN_C = len(context)

    global CONTEXT_CONTENT
    CONTEXT_CONTENT = context
    
    # SUMMARY
    #context = ""
    #for i, doc_id in enumerate(result["ids"]):
    #    context += "\n\n" + summarize_chunks(query, result["documents"][i])
    
    full_promt = promt.replace("<CONTEXT>", context).replace("<QUERY>", query).replace("\n\n", "\n")
    #print(f"DEBUG: promt len tokens: {len(full_promt) // 3}")
    global FULL_PROMT_LEN_C
    FULL_PROMT_LEN_C = len(full_promt)
    return full_promt

def run_gemma_one_query(query, model, tokenizer, n_chunks, question_n, application_id=None):

    #model, tokenizer, n_chunks = init_model(model_alias)
    
    if(N_CHUNKS != -1):
        n_chunks = N_CHUNKS
    global RANK_B
    RANK_B = time.perf_counter()
    promt = prepare_promt(query, n_chunks, question_n, application_id)
    global RANK_E
    RANK_E = time.perf_counter()
    messages = [    
    {
    "role": "user",
    "content": [{"type" : "text", "text" : promt,}]
    }]

    global REQUEST_B
    REQUEST_B = time.perf_counter()
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt = True, # Must add for generation
        return_tensors = "pt",
        tokenize = True,
        return_dict = True,
    ).to("cuda")

    outputs = model.generate(
        **inputs,
        max_new_tokens = 128, # Increase for longer outputs!
        temperature = MODEL_TEMPERATURE, top_p = 0.9, top_k = 40,
    )

    # decoded answer of the model
    generated_text = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True,
    )
    global REQUEST_E
    REQUEST_E = time.perf_counter()

    global INPUT_TOKEN_COUNT 
    INPUT_TOKEN_COUNT = inputs["input_ids"].shape[-1]
    
    generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    output_token_count = generated_ids.shape[-1]
    
    global OUTPUT_TOKEN_COUNT
    OUTPUT_TOKEN_COUNT = output_token_count
    
    return generated_text, query, promt

def run_phi_one_query(query, model, tokenizer, n_chunks, question_n, application_id=None):
    
    if(N_CHUNKS != -1):
        n_chunks = N_CHUNKS
        
    global RANK_B
    RANK_B = time.perf_counter()
    promt = prepare_promt(query, n_chunks, question_n, application_id=None)
    global RANK_E
    RANK_E = time.perf_counter()
    
    messages = [
    {
        "role": "user",
        "content": promt,
    }
    ]
    
    global REQUEST_B
    REQUEST_B = time.perf_counter()
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    ).to("cuda")

    encoded = tokenizer(
        tokenizer.decode(input_ids[0]),
        return_tensors="pt"
    ).to("cuda")

    outputs = model.generate(
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        max_new_tokens=32,
        temperature=MODEL_TEMPERATURE,
        top_p=0.90,
        top_k=40,
        do_sample=False,
        repetition_penalty=1.0, 
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id 
    )
    
    generated_ids = outputs[0, input_ids.shape[-1]:]
    generated_text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )
    global REQUEST_E
    REQUEST_E = time.perf_counter()

    global INPUT_TOKEN_COUNT 
    INPUT_TOKEN_COUNT = len(encoded["input_ids"][0])

    global OUTPUT_TOKEN_COUNT
    OUTPUT_TOKEN_COUNT = len(generated_ids)
    
    return generated_text, query, promt

def calc_accuracy(generated_text, app_id, i):

    json_questions = QAPair.get(app_id.strip())
    if not json_questions:
        return None

    json_key = f"{i}"
    json_question_text = json_questions.get(json_key)
    if not json_question_text:
        return None

    #print(f"json {generated_text} {json_question_text}")
    corpus = [generated_text, json_question_text]
    tfidf = vectorizer.fit_transform(corpus)
    sim = cosine_similarity(tfidf[0], tfidf[1])[0][0]

    return sim

def run_slm_many_queries(model, tokenizer, n_chunks, run, model_alias):
    with open(APPLICATION_LIST_PATH, "r", encoding="utf-8") as af:
        for j, app_id in enumerate(af, start=1):
            with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
                for i, question in enumerate(f, start=1):
                    
                    if(model_alias == "gemma"):
                        generated_text, query, promt = run_gemma_one_query(question.strip(), model, tokenizer, n_chunks, i, app_id.strip())

                        accuracy = calc_accuracy(generated_text, app_id, i)
                        throughput = OUTPUT_TOKEN_COUNT / (REQUEST_E-REQUEST_B+RANK_E-RANK_B) if (REQUEST_E-REQUEST_B+RANK_E-RANK_B) > 0 else 0
                        word_count = len(generated_text.split())
                        sentence_count = generated_text.count('.') + generated_text.count('!') + generated_text.count('?')
                        reading_ease = textstat.flesch_reading_ease(generated_text)
                        add_record_to_csv('metrics_gemma.csv', run=run, model=model_alias, application_id=app_id, 
                                         question_id=i, question=question, answer=generated_text, accuracy = accuracy,
                                         time_rag_s=RANK_E-RANK_B, time_res_s=REQUEST_E-REQUEST_B, 
                                         time_total_s=REQUEST_E-REQUEST_B+RANK_E-RANK_B, throughput=throughput, context_len_c=CONTEXT_LEN_C,
                                         full_promt_len_c=FULL_PROMT_LEN_C, word_count_answ=word_count, sentence_count = sentence_count, input_token_cout=INPUT_TOKEN_COUNT,
                                         output_token_count=OUTPUT_TOKEN_COUNT, total_tokens= INPUT_TOKEN_COUNT+OUTPUT_TOKEN_COUNT, reading_ease = reading_ease,
                                         answer_len_c=len(generated_text), chunk_size_c=CHUNK_SIZE,
                                         chunk_overlap_c=CHUNK_OVERLAP, full_promt = promt)
                        
                        print(f"r:{run} gemma App:{app_id} Q:{i}: A:{generated_text}")
                        
                    elif(model_alias == "phi"): 
                        
                        generated_text, query, promt = run_phi_one_query(question.strip(), model, tokenizer, n_chunks, i, app_id.strip())

                        print(f"PREJSON {app_id}, {i}")
                        accuracy = calc_accuracy(generated_text, app_id, i)
                        throughput = OUTPUT_TOKEN_COUNT / (REQUEST_E-REQUEST_B+RANK_E-RANK_B) if (REQUEST_E-REQUEST_B+RANK_E-RANK_B) > 0 else 0
                        word_count = len(generated_text.split())
                        sentence_count = generated_text.count('.') + generated_text.count('!') + generated_text.count('?')
                        reading_ease = textstat.flesch_reading_ease(generated_text)
                        add_record_to_csv('metrics_phi.csv', run=run, model=model_alias, application_id=app_id, 
                                         question_id=i, question=question, answer=generated_text, accuracy = accuracy,
                                         time_rag_s=RANK_E-RANK_B, time_res_s=REQUEST_E-REQUEST_B, 
                                         time_total_s=REQUEST_E-REQUEST_B+RANK_E-RANK_B, throughput=throughput, context_len_c=CONTEXT_LEN_C,
                                         full_promt_len_c=FULL_PROMT_LEN_C, word_count_answ=word_count, sentence_count = sentence_count, input_token_cout=INPUT_TOKEN_COUNT,
                                         output_token_count=OUTPUT_TOKEN_COUNT, total_tokens= INPUT_TOKEN_COUNT+OUTPUT_TOKEN_COUNT, reading_ease = reading_ease,
                                         answer_len_c=len(generated_text), chunk_size_c=CHUNK_SIZE,
                                         chunk_overlap_c=CHUNK_OVERLAP, full_promt = promt)
                        
                        print(f"r:{run} phi App:{app_id} Q:{i}: A:{generated_text}")
                        
                    else:
                        print("Incorrect model selected")               
            
def one_question(query, model_alias):
    model, tokenizer, n_chunks = init_model(model_alias)

    if(model_alias == "gemma"):
        generated_text, _, _ = run_gemma_one_query(query, model, tokenizer, n_chunks, 1)
        print(f"QUESTION: {query} \n ANSWER: {generated_text}")
    elif(model_alias == "phi"):   
        generated_text, _, _ = run_phi_one_query(query, model, tokenizer, n_chunks, 1)
        print(f"QUESTION: {query} \n ANSWER: {generated_text}")
    else:
        print("Incorrect model selected")
    
def questions_from_file(model_alias):
    model, tokenizer, n_chunks = init_model(model_alias)
    for i in range(1, 4): # num of runs
        run_slm_many_queries(model, tokenizer, n_chunks, i, model_alias)
    

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-init",
        action="store_true",
        help="Make chunks from files and init vector storage. Needs to be run once before usage."
    )

    parser.add_argument(
        "-one",
        action="store_true",
        help="To ask slm one question"
    )

    parser.add_argument(
        "-multi",
        action="store_true",
        help="Iteration through the list of questions from questions.txt"
    )

    parser.add_argument(
        "-m",
        type=str,
        help="Name of the SLM. gemma or phi"
    )
    
    parser.add_argument(
        "-q",
        type=str,
        help="Question to the model (required for -one only)"
    )

    args = parser.parse_args()

    if args.one:
        if not args.m or not args.q:
            parser.error("arguments -m (model name: gemma or phi) and -q (question) are required when using -one")

    if args.multi:
        if not args.m:
            parser.error("argument -m (model name: gemma or phi) is required when using -multi")

    if args.one and args.multi:
        parser.error("arguments -one and -multi cannot be used together")

    if args.init:
        populate_vector_storage()
    elif args.one:
        #one_question(args.q, args.m) # don't work
        pass
    elif args.multi:
        questions_from_file(args.m)
    else:
        parser.error("Please select argument")


if __name__ == "__main__":
    main()