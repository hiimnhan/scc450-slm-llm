import os
from google import genai
from google.genai import types
import pathlib
from typing import List

# Hardcoded API Key:
API_KEY = "AIzaSyAyKLxLvb_6y6w1AhIlVdHQxKKwinQSgC0"

# This is a test API call:
client = genai.Client(api_key=API_KEY)


# --- Setup and Utility Functions ---

def upload_multiple_files(client: genai.Client, file_paths: List[str]) -> List[types.File]:
    """Uploads multiple files to the Gemini API and returns a list of File objects."""
    uploaded_files = []
    print("Starting file uploads...")
    for path in file_paths:
        try:
            # Use pathlib.Path for easy path handling
            file_path = pathlib.Path(path)
            if not file_path.exists():
                print(f"File not found: {path}. Skipping.")
                continue

            # Upload the file
            uploaded_file = client.files.upload(
                file=file_path,
                # set the file name accordingly
                config=types.UploadFileConfig(display_name=file_path.name)
            )

            uploaded_files.append(uploaded_file)
            print(f"Uploaded file: {uploaded_file.display_name} (Name: {uploaded_file.name})")
        except Exception as e:
            print(f"Error uploading {path}: {e}")
            
    print("All files uploaded.")
    return uploaded_files

def cleanup_files(client: genai.Client, files_to_delete: List[types.File]):
    """Deletes the uploaded File objects to free up space and storage."""
    print("\nStarting file cleanup...")
    for file in files_to_delete:
        try:
            client.files.delete(name=file.name)
            print(f"Deleted file: {file.display_name}")
        except Exception as e:
            print(f"Error deleting file {file.name}: {e}")
    print("Cleanup complete.")

# --- Main API Logic ---

def analyze_pdfs_with_gemini(pdf_paths: List[str], prompt_text: str):
    """Initializes the client, uploads files, generates content, and cleans up."""
    try:
        # Client will automatically pick up the GEMINI_API_KEY environment variable.
        # Ensure 'pip install google-genai' is run first.
        client = genai.Client(api_key=API_KEY)
    except Exception as e:
        print("Error initializing Gemini client. Make sure your GEMINI_API_KEY environment variable is set.")
        print(f"Details: {e}")
        return

    # 1. Upload the files
    uploaded_files = upload_multiple_files(client, pdf_paths)
    
    if not uploaded_files:
        print("No files were successfully uploaded. Exiting.")
        return

    # 2. Prepare contents for the model
    # The contents list will contain the File objects and the text prompt.
    contents = uploaded_files + [prompt_text]

    # 3. Call the model
    print("\nCalling Gemini-2.5-Flash model...")
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=contents
        )
        
        # 4. Display the response
        print("\n--- Gemini Response ---")
        print(response.text)
        print("-----------------------")
        
    except Exception as e:
        print(f"Error generating content: {e}")

    # 5. Cleanup
    # cleanup_files(client, uploaded_files)


# --- Execution Block ---

if __name__ == "__main__":
    # IMPORTANT: Replace these with the actual paths to your local PDF files.
    # For this example to run, you must have these files in your local directory.
    local_pdf_files = [
        "Trafford Validation-checklist.pdf"
    ]
    
    # Define the prompt you want to ask the model about the files
    analysis_prompt = (
        "print out 'hello world'"
    )
    

    analyze_pdfs_with_gemini(local_pdf_files, analysis_prompt)