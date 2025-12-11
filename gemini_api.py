import time
from google import genai
from google.genai import types
import textstat
import os
import csv
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# ---------------------------
# CONFIGURATION
# ---------------------------
# Global variable setup
API_KEY = "" # hardcoded API key
MODEL = "gemini-2.5-flash"  # A versatile and fast multimodal model
N_REPEATS = 3
REFERENCE_ANSWER = ""  # to be replaced with expected answer
SYSTEM_INSTRUCTION = "You are a helpful and concise assistant. Only use the information provided in the file(s) to answer the question, unless asked otherwise."

# Pricing for gemini-2.5-flash (Example: pricing may vary, check current docs)
# Input: $0.30/1M tokens -> $0.00030/1K tokens
COST_INPUT_PER_1K = 0.30 / 1000
# Output: $2.50/1M tokens -> $0.00250/1K tokens
COST_OUTPUT_PER_1K = 2.50 / 1000

# Paths to files to be uploaded
local_pdf_files = [
    "Testing Files PDF/109428-CND-22.pdf",
    "Testing Files PDF/109509-CND-22.pdf",
    "Testing Files PDF/109552-S211-22.pdf",
    "Testing Files PDF/109435-CPL-22.pdf",	
    "Testing Files PDF/109516-NMA-22.pdf",	
    "Testing Files PDF/109578-FUL-22.pdf",
    "Testing Files PDF/109448-CND-22.pdf",	
    "Testing Files PDF/109533-TPO-22.pdf",	
    "Testing Files PDF/109593-TPO-22.pdf",
    "Testing Files PDF/109461-TPO-22.pdf",	
    "Testing Files PDF/109546-CPL-22.pdf",	
    "Testing Files PDF/109594-S211-22.pdf",
    "Testing Files PDF/109479-CND-22.pdf",	
    "Testing Files PDF/109547-CPL-22.pdf",
    "Testing Files PDF/109611-TPO-22.pdf",
    "Testing Files PDF/109496-S211-22.pdf",	
    "Testing Files PDF/109549-S211-22.pdf",	
    "Testing Files PDF/109623-S211-22.pdf",
    "Testing Files PDF/109508-NMA-22.pdf",	
    "Testing Files PDF/109551-S211-22.pdf"
]

# Imported question-answer pair as a dictionary

sheet = {
 "application_summaries": [
   {
     "folder_id": "109428",
     "questions": {
       "Question 1": "Application for approval of details reserved by condition",
       "Question 2": "No",
       "Question 3": "No",
       "Question 4": "No",
       "Question 5": "No"
     }
   },
   {
     "folder_id": "109435",
     "questions": {
       "Question 1": "Lawful development certificates (existing and proposed)",
       "Question 2": "No",
       "Question 3": "Yes",
       "Question 4": "No",
       "Question 5": "No"
     }
   },
   {
     "folder_id": "109448",
     "questions": {
       "Question 1": "Application for approval of details reserved by condition",
       "Question 2": "No",
       "Question 3": "No",
       "Question 4": "No",
       "Question 5": "No"
     }
   },
   {
     "folder_id": "109461",
     "questions": {
       "Question 1": "Application for tree works",
       "Question 2": "No",
       "Question 3": "No",
       "Question 4": "Yes",
       "Question 5": "No"
     }
   },
   {
     "folder_id": "109479",
     "questions": {
       "Question 1": "Application for Approval of Details Reserved by Condition",
       "Question 2": "No",
       "Question 3": "No",
       "Question 4": "No",
       "Question 5": "No"
     }
   },
   {
     "folder_id": "109496",
     "questions": {
       "Question 1": "Application for Tree Works",
       "Question 2": "No",
       "Question 3": "Yes",
       "Question 4": "Yes",
       "Question 5": "No"
     }
   },
   {
     "folder_id": "109508",
     "questions": {
       "Question 1": "Application for a Non-Material Amendment Following a Grant of Planning Permission",
       "Question 2": "No",
       "Question 3": "No",
       "Question 4": "No",
       "Question 5": "No"
     }
   },
   {
     "folder_id": "109509",
     "questions": {
       "Question 1": "Application for Approval of Details Reserved by Condition",
       "Question 2": "No",
       "Question 3": "No",
       "Question 4": "No",
       "Question 5": "No"
     }
   },
   {
     "folder_id": "109516",
     "questions": {
       "Question 1": "Application for a Non-Material Amendment Following a Grant of Planning Permission",
       "Question 2": "No",
       "Question 3": "No",
       "Question 4": "No",
       "Question 5": "No"
     }
   },
   {
     "folder_id": "109533",
     "questions": {
       "Question 1": "Application for Tree Works",
       "Question 2": "No",
       "Question 3": "No",
       "Question 4": "Yes",
       "Question 5": "No"
     }
   },
   {
     "folder_id": "109546",
     "questions": {
       "Question 1": "Lawful development certificates",
       "Question 2": "No",
       "Question 3": "Yes",
       "Question 4": "No",
       "Question 5": "No"
     }
   },
   {
     "folder_id": "109547",
     "questions": {
       "Question 1": "Lawful development certificates",
       "Question 2": "No",
       "Question 3": "Yes",
       "Question 4": "No",
       "Question 5": "No"
     }
   },
   {
     "folder_id": "109549",
     "questions": {
       "Question 1": "Application for Tree Works",
       "Question 2": "No",
       "Question 3": "Yes",
       "Question 4": "Yes",
       "Question 5": "No"
     }
   },
   {
     "folder_id": "109551",
     "questions": {
       "Question 1": "Application for Tree Works",
       "Question 2": "No",
       "Question 3": "Yes",
       "Question 4": "Yes",
       "Question 5": "No"
     }
   },
   {
     "folder_id": "109552",
     "questions": {
       "Question 1": "Application for Tree Works",
       "Question 2": "No",
       "Question 3": "Yes",
       "Question 4": "Yes",
       "Question 5": "No"
     }
   },
   {
     "folder_id": "109578",
     "questions": {
       "Question 1": "Application for planning permission (full)",
       "Question 2": "Yes",
       "Question 3": "No",
       "Question 4": "No",
       "Question 5": "No"
     }
   },
   {
     "folder_id": "109593",
     "questions": {
       "Question 1": "Application for Tree Works",
       "Question 2": "No",
       "Question 3": "Yes",
       "Question 4": "Yes",
       "Question 5": "No"
     }
   },
   {
     "folder_id": "109594",
     "questions": {
       "Question 1": "Application for Tree Works",
       "Question 2": "No",
       "Question 3": "Yes",
       "Question 4": "Yes",
       "Question 5": "No"
     }
   },
   {
     "folder_id": "109611",
     "questions": {
       "Question 1": "Application for Tree Works",
       "Question 2": "No",
       "Question 3": "Yes",
       "Question 4": "Yes",
       "Question 5": "No"
     }
   },
   {
     "folder_id": "109623",
     "questions": {
       "Question 1": "Application for Tree Works",
       "Question 2": "No",
       "Question 3": "Yes",
       "Question 4": "Yes",
       "Question 5": "No"
     }
   }
 ]
}

reference_sheet = sheet["application_summaries"] # Unwrap the dictionary

# Define the prompt you want to ask the model about the files
analysis_prompt = {# "Answer each of the following 5 questions regarding the single pdf file."
    "Question 1": "What is the development type of this application? Answer in a few words, not in sentence.",
    "Question 2": "Does this application require demolition within a conservation area? Answer YES or NO only. Do not add any other comment.",
    "Question 3": "Does the application include a site location plan? Answer YES or NO only. Do not add any other comment.",
    "Question 4": "Does this application involve works to trees? Answer YES or NO only. Do not add any other comment.",
    "Question 5": "Does the proposal include any new buildings or extensions? Answer YES or NO only. Do not add any other comment."
}

# Initialize Gemini Client and Embedding Model
client = genai.Client(api_key=API_KEY)
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# ---------------------------
# INPUT & FILE UPLOAD
# ---------------------------

# List of local file paths (e.g., PDFs, JPEGs)


pdf_files = local_pdf_files # e.g., ["/path/to/doc.pdf", "/path/to/image.jpg"]
uploaded_files = [] # Lists of successfully uploaded files
application_number = [] # List of names of uploaded files to be referred to later on model calls

# Upload files to the Gemini API File Service (temporary storage)
print("Uploading files...")
for file_path in pdf_files:

    # Use client.files.upload, which returns a file object
    uploaded_file = client.files.upload(file=file_path)
    uploaded_files.append(uploaded_file)
    application_number.append(file_path[18:24])
    print(f"Uploaded: {uploaded_file.name} (MIME: {uploaded_file.mime_type})")


def compare_answer(correct_answer, answer): # answer is extracted from model response 
    answer_embedding = embed_model.encode([answer])[0]
    reference_embedding = embed_model.encode([correct_answer])[0]
    accuracy = cosine_similarity([answer_embedding], [reference_embedding])[0][0]
    return accuracy


# ---------------------------
# FUNCTION: RUN MODEL ONCE
# ---------------------------

def evaluate_once(contents_list: list, system_instruction: str,
                  which_application: str, question_number: str) -> dict:
    """
    Runs the model once, collects metrics, and cleans up uploaded files.
    """
    start_time = time.time()
    result = {}

    # Call the Gemini API
    response = client.models.generate_content(
        model=MODEL,
        contents=contents_list,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction
        )
    )

    end_time = time.time()
    answer = response.text
    
    # Extract token usage from the response
    usage = response.usage_metadata
    prompt_tokens = usage.prompt_token_count
    completion_tokens = usage.candidates_token_count
    total_tokens = usage.total_token_count
    
    # Calculate cost
    total_cost = (prompt_tokens / 1000 * COST_INPUT_PER_1K) + \
                    (completion_tokens / 1000 * COST_OUTPUT_PER_1K)
    
    # Calculate latency and throughput
    latency = end_time - start_time
    throughput = completion_tokens / latency if latency > 0 else 0

    # Run text analysis metrics (textstat is model-agnostic)
    word_count = textstat.lexicon_count(answer, removepunct=True)
    char_count = textstat.char_count(answer)
    sentence_count = textstat.sentence_count(answer)
    flesch_reading_ease = textstat.flesch_reading_ease(answer)
    
    # Semantic similarity is calculated later after all runs
    
    # lookup the correct answer in reference sheet by folder_id and question_number
    correct_answer = ''
    for i in range(len(reference_sheet)):
        if reference_sheet[i]["folder_id"] == which_application:
            correct_answer = reference_sheet[i]["questions"][question_number]

    accuracy = compare_answer(correct_answer, answer)

    result.update({
        "answer": answer,
        "latency": latency,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "total_cost": total_cost,
        "throughput": throughput,
        "word_count": word_count,
        "char_count": char_count,
        "sentence_count": sentence_count,
        "flesch_reading_ease": flesch_reading_ease,
        "accuracy": accuracy,
        "consistency": 0.0, # Placeholder, calculated later
    })
    
    return result



# --------------------------
# EXECUTE REPEATED RUNS
# --------------------------

# Only feed one uploaded file paired with one question in prompt per model call
all_results = []
for i in range(N_REPEATS):
    for j in range(len(uploaded_files)):
        one_file = uploaded_files[j]
        which_application = application_number[j]
        for q_number, one_question in analysis_prompt.items():
            
            contents = [one_question] + [one_file]
            print(f"\n--- Running evaluation {i+1}/{N_REPEATS}...")
            # Pass the full content list and system instruction
            result = evaluate_once(contents, SYSTEM_INSTRUCTION, which_application, q_number)
            result["run"] = i + 1
            all_results.append(result)

# --------------------------
# CLEANUP FILES
# --------------------------
print("\nCleaning up uploaded files...")
for uploaded_file in uploaded_files:
    try:
        client.files.delete(name=uploaded_file.name)
        print(f"Deleted file: {uploaded_file.name}")
    except Exception as e:
        print(f"Could not delete file {uploaded_file.name}: {e}")

# --------------------------
# CALCULATE ACCURACY & CONSISTENCY
# --------------------------


if REFERENCE_ANSWER:
    # 1. Embed all answers and the reference answer
    answers = [r['answer'] for r in all_results]
    all_texts = [REFERENCE_ANSWER] + answers
    
    # Encode all texts at once
    print("Calculating embeddings for similarity...")
    all_embeddings = embed_model.encode(all_texts)
    reference_embedding = all_embeddings[0]
    answer_embeddings = all_embeddings[1:]

    # 2. Calculate Accuracy (Similarity to Reference Answer)
    reference_embedding = reference_embedding.reshape(1, -1)
    accuracy_scores = [
        cosine_similarity(reference_embedding, ans.reshape(1, -1))[0][0] 
        for ans in answer_embeddings
    ]
    
    # 3. Calculate Consistency (Average pairwise similarity between answers)
    consistency_scores = []
    for i in range(N_REPEATS):
        sims = []
        for j in range(N_REPEATS):
            if i != j:
                # Calculate similarity between answer i and answer j
                sim = cosine_similarity(
                    answer_embeddings[i].reshape(1, -1), 
                    answer_embeddings[j].reshape(1, -1)
                )[0][0]
                sims.append(sim)
        consistency_scores.append(sum(sims) / (N_REPEATS - 1) if N_REPEATS > 1 else 1.0)
        
    # 4. Update results with scores
    for r, acc, cons in zip(all_results, accuracy_scores, consistency_scores):
        r["accuracy"] = acc
        r["consistency"] = cons
else:
    print("\nSkipping accuracy and consistency calculation: REFERENCE_ANSWER is empty.")


    
# --------------------------
# SAVE TO CSV
# --------------------------
csv_file = "llm-evaluation-gemini.csv"
print(f"\nSaving results to {csv_file}...")

with open(csv_file, mode="w", newline='', encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Run","Answer","Latency (s)","Prompt Tokens","Completion Tokens","Total Tokens",
        "Total Cost ($)","Throughput (tokens/s)","Word Count","Character Count","Sentence Count",
        "Flesch Reading Ease","Accuracy","Consistency"
    ])

    for r in all_results:
        writer.writerow([
            r["run"], r["answer"], f"{r['latency']:.4f}", r["prompt_tokens"], r["completion_tokens"], r["total_tokens"],
            f"{r['total_cost']:.6f}", f"{r['throughput']:.2f}", r["word_count"], r["char_count"], r["sentence_count"],
            f"{r['flesch_reading_ease']:.2f}", f"{r['accuracy']:.4f}", f"{r['consistency']:.4f}"
        ])

print("Evaluation complete.")




