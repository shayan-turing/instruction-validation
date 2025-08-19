from sentence_transformers import SentenceTransformer
import faiss
import json
import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS
from anthropic import Anthropic
import re
from dotenv import load_dotenv
load_dotenv()

if os.environ.get('RENDER'):
    # Production on Render - use temp directories
    UPLOAD_FOLDER = "/tmp/uploads"
    VECTOR_FOLDER = "/tmp/vectorized"
else:
    # Local development
    UPLOAD_FOLDER = "uploads"
    VECTOR_FOLDER = "vectorized"
    
ALLOWED_EXTENSIONS = {"json","md","txt"}

def ensure_directories():
    try:
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(VECTOR_FOLDER, exist_ok=True)
        print(f"Directories created/verified:")
        print(f"  Upload folder: {UPLOAD_FOLDER} (exists: {os.path.exists(UPLOAD_FOLDER)})")
        print(f"  Vector folder: {VECTOR_FOLDER} (exists: {os.path.exists(VECTOR_FOLDER)})")
        
        # Test write permissions
        test_file = os.path.join(UPLOAD_FOLDER, 'test_write.txt')
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        print(f"  Upload folder is writable: True")
        
    except Exception as e:
        print(f"Error creating directories: {str(e)}")
        raise

# Create directories on startup
ensure_directories()

app = Flask(__name__)
CORS(app)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER



api_key = os.environ.get("ANTHROPIC_API_KEY")
hf_token = hf_token = os.environ.get("HF_TOKEN")
anthropic_client = Anthropic(api_key=api_key)

embedder = SentenceTransformer("BAAI/bge-large-en-v1.5", use_auth_token=hf_token)


# Helper function to call Claude
def call_claude(prompt, model="claude-sonnet-4-20250514", max_tokens=4000, temperature=0.1):

    client = anthropic_client
    
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    return response.content[0].text


def allowed_files(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS



def chunk_text(text, chunk_size=1000, overlap=200):
    """Split text into overlapping chunks for better retrieval"""
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = min(start + chunk_size, text_len)
        
        # Try to break at sentence end if possible
        if end < text_len:
            # Look for sentence endings within the last 100 characters
            last_period = text.rfind('.', start + chunk_size - 100, end)
            if last_period > start:
                end = last_period + 1
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap if end < text_len else text_len
    
    return chunks

def parse_markdown(file_path):
    """Parse markdown file and extract sections"""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Split by headers and create structured chunks
    sections = []
    current_section = {"title": "", "content": ""}
    
    lines = content.split('\n')
    for line in lines:
        # Check if line is a header
        if line.startswith('#'):
            # Save previous section if it has content
            if current_section["content"].strip():
                sections.append(current_section)
            
            # Start new section
            header_level = len(line) - len(line.lstrip('#'))
            title = line.lstrip('# ').strip()
            current_section = {
                "title": title,
                "content": "",
                "level": header_level
            }
        else:
            # Add to current section content
            current_section["content"] += line + '\n'
    
    # Add the last section
    if current_section["content"].strip():
        sections.append(current_section)
    
    # Convert sections to text chunks
    texts = []
    metadata = []
    
    for section in sections:
        title = section.get("title", "")
        content = section.get("content", "").strip()
        level = section.get("level", 1)
        
        if not content:
            continue
        
        # For large sections, chunk them further
        if len(content) > 1500:
            chunks = chunk_text(content, chunk_size=1000, overlap=200)
            for i, chunk in enumerate(chunks):
                full_text = f"# {title}\n\n{chunk}" if title else chunk
                texts.append(full_text)
                metadata.append({
                    "title": title,
                    "section": f"{title}_part_{i+1}" if title else f"section_part_{i+1}",
                    "level": level,
                    "content": chunk,
                    "type": "markdown_section"
                })
        else:
            full_text = f"# {title}\n\n{content}" if title else content
            texts.append(full_text)
            metadata.append({
                "title": title,
                "section": title or "content",
                "level": level,
                "content": content,
                "type": "markdown_section"
            })
    
    return texts, metadata

def parse_text_file(file_path):
    """Parse plain text file"""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Split into chunks
    chunks = chunk_text(content, chunk_size=1000, overlap=200)
    
    texts = []
    metadata = []
    
    for i, chunk in enumerate(chunks):
        texts.append(chunk)
        metadata.append({
            "chunk_id": i + 1,
            "content": chunk,
            "type": "text_chunk"
        })
    
    return texts, metadata

def build_index(input_file, text_key, output_prefix):
    try:
        print(f"Building index for file: {input_file}")
        
        # Determine file type and parse accordingly
        file_extension = input_file.lower().split('.')[-1]
        
        if file_extension == 'json':
            print(f"Processing JSON file with text key: {text_key}")
            
            # Read and parse JSON file
            with open(input_file, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    print(f"Successfully loaded JSON with {len(data)} items")
                except json.JSONDecodeError as e:
                    print(f"JSON decode error: {str(e)}")
                    return False, f"Invalid JSON format: {str(e)}"
        
            # Extract texts from JSON
            texts = []
            metadata = []
            
            for i, item in enumerate(data):
                if not isinstance(item, dict):
                    print(f"Item {i} is not a dictionary: {type(item)}")
                    continue
                    
                text_val = item.get(text_key, "")
                if isinstance(text_val, str) and text_val.strip():
                    texts.append(text_val.strip())
                    metadata.append(item)
                else:
                    print(f"Item {i} has invalid text value for key '{text_key}': {text_val}")
            
            if not texts:
                available_keys = list(data[0].keys()) if data else []
                print(f"Available keys in first item: {available_keys}")
                return False, f"No valid text entries found for key '{text_key}'. Available keys: {available_keys}"
        
        elif file_extension == 'md':
            print("Processing Markdown file")
            texts, metadata = parse_markdown(input_file)
            
        elif file_extension == 'txt':
            print("Processing text file")
            texts, metadata = parse_text_file(input_file)
            
        else:
            return False, f"Unsupported file type: {file_extension}"
        
        print(f"Extracted {len(texts)} text entries")
        
        if not texts:
            return False, "No valid text entries found in file"
        
        # Generate embeddings
        print("Generating embeddings...")
        try:
            embeddings = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            print(f"Generated embeddings shape: {embeddings.shape}")
        except Exception as e:
            print(f"Error generating embeddings: {str(e)}")
            return False, f"Error generating embeddings: {str(e)}"
        
        # Create FAISS index
        print("Creating FAISS index...")
        try:
            dim = embeddings.shape[1]
            index = faiss.IndexFlatL2(dim)
            index.add(embeddings)
            print(f"Created index with {index.ntotal} vectors")
        except Exception as e:
            print(f"Error creating FAISS index: {str(e)}")
            return False, f"Error creating FAISS index: {str(e)}"
        
        # Save files
        try:
            faiss_path = os.path.join(VECTOR_FOLDER, f"{output_prefix}.faiss")
            texts_path = os.path.join(VECTOR_FOLDER, f"{output_prefix}_texts.json")
            meta_path = os.path.join(VECTOR_FOLDER, f"{output_prefix}_meta.json")
            
            faiss.write_index(index, faiss_path)
            
            with open(texts_path, "w", encoding="utf-8") as f:
                json.dump(texts, f, indent=2, ensure_ascii=False)
            
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            print(f"Saved index files: {faiss_path}, {texts_path}, {meta_path}")
            
        except Exception as e:
            print(f"Error saving index files: {str(e)}")
            return False, f"Error saving index files: {str(e)}"
    
        return True, f"Indexed {len(texts)} entries successfully"
        
    except Exception as e:
        print(f"Unexpected error in build_index: {str(e)}")
        return False, f"Unexpected error: {str(e)}"
    
    
    
def delete_index(prefix):
    files = [
        f"{prefix}.faiss",
        f"{prefix}_texts.json",
        f"{prefix}_meta.json"
    ]
    for file in files:
        path = os.path.join(VECTOR_FOLDER, file)
        if os.path.exists(path):
            os.remove(path)
            
def load_index(prefix):
    index_path = os.path.join(VECTOR_FOLDER, f"{prefix}.faiss")
    text_path = os.path.join(VECTOR_FOLDER, f"{prefix}_texts.json")
    if not os.path.exists(index_path) or not os.path.exists(text_path):
        return None, None
    index = faiss.read_index(index_path)
    with open(text_path) as f:
        texts = json.load(f)
    return index, texts

def retrieve(index, texts, query, top_k=3):
    query_vec = embedder.encode([query], convert_to_numpy=True)
    D, I = index.search(query_vec, top_k)
    return [texts[i] for i in I[0]]

def run_search(query, index_name=None, top_k=3):
    if not query:
        return {"status":"error", "message":"Query is required"}
    if index_name:
        index,texts = load_index(index_name)
        if not index:
            return {"status":"error","message":f"Index '{index_name}' not found"}
        results = retrieve(index,texts,query,top_k)
        return {"status":"success", "results":results}
    results = {}
    for file in os.listdir(VECTOR_FOLDER):
        if file.endswith('.faiss'):
            prefix = file.replace(".faiss","")
            index,texts = load_index(prefix)
            if index:
                results[prefix] = retrieve(index,texts,query,top_k)
    return {"status":"success","results":results}



# Updated upload route that doesn't require text_key for non-JSON files
@app.route("/upload", methods=["POST"])
def upload_file():
    try:
        print("Upload request received")
        
        if "file" not in request.files:
            print("No file in request")
            return jsonify({"status": "error", "message": "No file uploaded"}), 400
        
        file = request.files["file"]
        text_key = request.form.get("text_key", "text")  # Default for JSON files
        
        print(f"File received: {file.filename}")
        print(f"Text key: {text_key}")
        
        if file.filename == "":
            print("Empty filename")
            return jsonify({"status": "error", "message": "No selected file"}), 400
        
        if not file or not allowed_files(file.filename):
            print("Invalid file type or no file")
            return jsonify({"status": "error", "message": "Invalid file type"}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        
        try:
            file.save(file_path)
            print(f"File saved to: {file_path}")
        except Exception as e:
            print(f"Error saving file: {str(e)}")
            return jsonify({"status": "error", "message": f"Error saving file: {str(e)}"}), 500
        
        # Build index
        prefix = filename.split(".", 1)[0]
        print(f"Building index with prefix: {prefix}")
        
        try:
            success, msg = build_index(file_path, text_key, prefix)
            print(f"Index build result: {success}, message: {msg}")
        except Exception as e:
            print(f"Error building index: {str(e)}")
            if os.path.exists(file_path):
                os.remove(file_path)
            return jsonify({"status": "error", "message": f"Error processing file: {str(e)}"}), 500
        
        if not success:
            print(f"Index building failed: {msg}")
            if os.path.exists(file_path):
                os.remove(file_path)
            return jsonify({"status": "error", "message": msg}), 400
        
        print(f"Upload successful: {msg}")
        return jsonify({"status": "success", "message": msg, "file": filename}), 200
        
    except Exception as e:
        print(f"Unexpected error in upload: {str(e)}")
        return jsonify({"status": "error", "message": f"Unexpected error: {str(e)}"}), 500
    
@app.route("/files",methods=["GET"])
def list_files():
    files = os.listdir(UPLOAD_FOLDER)
    return jsonify({"status":"success","files":files})

@app.route("/delete/<filename>", methods=["DELETE"])
def delete_file(filename):
    file_path = os.path.join(UPLOAD_FOLDER,filename)
    prefix = filename.rsplit(".",1)[0]
    if os.path.exists(file_path):
        os.remove(file_path)
        delete_index(prefix)
        return jsonify({"status":"success","message":f"{filename} and index deleted"})
    else:
        return jsonify({"status":"error","message":"File not found"}),400

@app.route("/search", methods=["POST"])
def search():
    data = request.get_json()
    query = data.get("query")
    index_name = data.get("index_name")
    top_k = int(data.get("top_k", 3))

    result = run_search(query, index_name=index_name, top_k=top_k)

    if result.get("status") == "error":
        return jsonify(result), 400

    prompt = f"""
    You are a policy compliance adviser. 
    Reframe these policies into simple, clear sentences.

    ⚠️ STRICT INSTRUCTION: Output ONLY valid JSON in this format:
    {{
      "results": [
        "policy 1 reformulated",
        "policy 2 reformulated",
        ...
      ],
      "status": "success"
    }}

    Policies for reference:
    {result}
    """

    raw_response = call_claude(prompt)

    # --- Extract only JSON ---
    try:
        match = re.search(r"\{.*\}", raw_response, re.DOTALL)
        if match:
            policy_result = json.loads(match.group(0))
        else:
            policy_result = {"results": [], "status": "error", "message": "No JSON found"}
    except Exception as e:
        policy_result = {"results": [], "status": "error", "message": str(e)}

    return jsonify(policy_result)


            
@app.route("/validate_instruction", methods=["POST"])
def validate_instruction():
    data = request.get_json()
    instruction = data.get("instruction")
    index_name = data.get("index_name")
    top_k = int(data.get("top_k", 3))

    if not instruction:
        return jsonify({"error": "Missing 'instruction' in request"}), 400

    search_result = run_search(instruction, index_name=index_name, top_k=top_k)
    if search_result.get("status") != "success":
        return jsonify({"error":"Search Failed", "details":search_result}),500
    context_results = search_result["results"]
    


    prompt = f"""
You are a policy compliance checker and an instruction validator.

Evaluation Criteria
1. USER-FACING
a. The instruction must be directed at an end user in natural, human-centered language.
b. It must not read like a direct command to a system, developer, or internal process.
c. Prefer “you” or “you want to…” phrasing.
d. Avoid technical commands like “delete invoice 42,” “POST to API endpoint,” or “run SQL query.”

2. OUTPUT/GOAL ORIENTED (not procedural)
a. The instruction must specify the exact outcome or solution the task is designed to achieve, not how to do it step-by-step.
b. Avoid “spoon-feeding” the system with exact process steps (looping, filtering, database operations, etc.).

3. SINGLE, UNAMBIGUOUS OUTCOME
a. One overarching outcome – It expresses a single desired result instead of multiple separate actions.
b. User-centered – Written in second person (“you want to…”) so it’s directed at the end user.
c. Context-preserving – All relevant details (time, people, conditions, constraints) are included, but only as supporting context to achieve that one outcome.
d. Clarity & measurability – The goal is specific enough that you can verify when it has been achieved.
e. No competing intents – Even if there are multiple steps, they are framed as sub-parts of one larger objective, not as separate goals.
Example:
Not a single goal (ambiguous): “Fix the bulb, create an alert, and schedule automation.”
Single unambiguous goal: “You want to ensure reliable bulb management by fixing the malfunction, logging the alert acknowledged by David Navarro, and setting up a daily shutdown routine.”

 
4. POLICY COMPLIANT
a. The instruction must follow the given policy and not introduce extra constraints that could conflict.
b. If it violates policy or imposes unrelated restrictions, it fails.

Pass/Fail Decision Tree
Is it user-facing? → No = FAIL.
Is it goal-oriented (not process)? → No = FAIL.
Does it have one clear outcome? → No = FAIL.
Is it policy compliant? → No = FAIL.
If all 4 pass → OVERALL: PASS.

improved_instruction:
a. The improved instruction should tell how to frame sentence so that single outcome is achieved.
b. Do not omit any detils that is given in the instruction.
c. Your job is to modify the instruction in such a way that all the checks passes, and the details in instruction are also not lost also keep in mind that the instruction should be compact.
d. Also keep in mind all the compliances while modifying the instruction like user_facing,output_oriented, single_outcome, policy_compliant, overall.

Policy rules:
{context_results}

Good Examples:
You are James Shawn (jamesshawn@gmail.com). On 2025-08-07, you want to add a 100,000 EUR subscription to the 'Emerging Markets Equity Fund' for investor Lawson-Edwards, assign it to yourself, and send them a subscription update email. You also want to create an invoice for half the subscription amount with due date 2025-08-31 and send an alert email for it.

Bad Examples:
You are Natasha Hickman (email: natashahickman@protonmail.com), an administrator who needs to handle a comprehensive commitment management scenario for multiple investors. First, you need to verify your identity and then check if investor ID 15 has any existing commitments for fund ID 25. If no commitment exists, you need to create a new commitment for 500,000 GBP with a commitment date of September 10, 2025. Then, retrieve all commitments for this investor to verify the creation. Next, you need to update the commitment amount to 750,000 GBP due to increased investor interest. After the update, check the commitment fulfillment status and calculate the fulfillment percentage. Additionally, create another commitment for investor ID 30 to fund ID 25 for 300,000 GBP with a commitment date of September 15, 2025. Retrieve all commitments for fund ID 25 to see both commitments. However, investor ID 30 has decided to withdraw, so you need to delete their commitment. Finally, generate a holding report for fund ID 25 for investor ID 15 with a report date and export period end date of September 25, 2025, and send an email notification of type alert to investor ID 15 about their updated commitment status.

Instruction:
{instruction}

Decide if the instruction is VALID or INVALID according to the rules.
Respond ONLY in format given below:
{{
  "user_facing": {{
    "result": "PASS/FAIL",
    "explanation": "..."
  }},
  "output_oriented": {{
    "result": "PASS/FAIL", 
    "explanation": "..."
  }},
  "single_outcome": {{
    "result": "PASS/FAIL",
    "explanation": "..."
  }},
  "policy_compliant": {{
    "result": "PASS/FAIL",
    "explanation": "..."
  }},
  "overall": {{
    "result": "PASS/FAIL",
    "summary": "...",
    "improved_instruction": ["...", "...", "..."]
  }}
}}
"""

    result_text = call_claude(prompt)
    

    if result_text.startswith("```"):
        result_text = result_text.strip("`").lstrip("json").strip()

    try:
        validation_data = json.loads(result_text)
    except json.JSONDecodeError as e:
        return jsonify({
            "error": "Invalid JSON from model",
            "raw_output": result_text,
            "details": str(e)
        }), 500

    return jsonify({"Validation": validation_data})

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))  # Render sets $PORT
    app.run(host="0.0.0.0", port=port)
