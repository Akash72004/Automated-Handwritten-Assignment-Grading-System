from typing import Dict, List, Any
import os, re, base64, functools, traceback

from pdf2image import convert_from_path, exceptions
from PIL import Image # type: ignore

# NLP and Grading specific imports
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize, sent_tokenize

# Ensure necessary NLTK data is downloaded (run once)
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')

import bert_score

from openai import OpenAI, types

OPENROUTER_API_KEY = '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' # Replace default if needed
VL_MODEL_NAME = "meta-llama/llama-4-maverick:free"
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
# --- Helper Functions ---
def allowed_file(filename):
    """Check if a file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'pdf', 'png', 'jpg', 'jpeg'}
def pdf_to_images(pdf_path, output_folder="images"):
    """Convert PDF pages to image files."""
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at {pdf_path}")
        return []
    try:
        os.makedirs(output_folder, exist_ok=True)
        print(f"Converting PDF '{os.path.basename(pdf_path)}' to images in '{output_folder}'...")
        # Note: Explicitly passing poppler_path might be needed on some systems
        # images = convert_from_path(pdf_path, poppler_path=POPPLER_PATH)
        images = convert_from_path(pdf_path)
        image_paths = []
        for i, image in enumerate(images):
            path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(pdf_path))[0]}_page_{i + 1}.jpg")
            image.save(path, "JPEG")
            image_paths.append(path)
        print(f"Converted PDF to {len(image_paths)} image(s).")
        return image_paths
    except exceptions.PDFInfoNotInstalledError:
        print("ERROR: pdfinfo command not found. Ensure Poppler is installed and in your PATH. See: https://pdf2image.readthedocs.io/en/latest/installation.html")
        return []
    except Exception as e:
        print(f"Error converting PDF '{pdf_path}' to images: {e}")
        print(traceback.format_exc())
        return []

@functools.lru_cache(maxsize=128)
def get_synonyms(word):
    """Get synonyms for a given word with caching for performance."""
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().replace("_", " "))
    return synonyms


def similar(a, b):
    """Check if two strings are similar using multiple metrics."""
    a, b = a.lower(), b.lower()

    # Direct match or substring check
    if a == b or a in b or b in a:
        return True

    # Character similarity (Jaccard similarity) - prevent division by zero
    a_chars, b_chars = set(a), set(b)
    intersection = len(a_chars.intersection(b_chars))
    union = len(a_chars.union(b_chars))
    if union == 0:
        return True if intersection == 0 else False  # Both empty strings are similar

    return intersection / union > 0.7  # Threshold from original code

def extract_text_from_question_paper(image_path, client : OpenAI, VL_MODEL_NAME : str):
    """
    Use VL model to extract text from question paper image with specific formatting.
    Returns structured question paper text or None on error.
    """
    if not client:
        print("ERROR: OpenRouter client not initialized. Cannot call API.")
        return None
    if not os.path.exists(image_path):
        print(f"ERROR: Question paper image not found at {image_path}")
        return None

    print(f"Extracting text from Question Paper: {os.path.basename(image_path)} using {VL_MODEL_NAME}...")
    try:
        with open(image_path, "rb") as img_file:
            base64_image = base64.b64encode(img_file.read()).decode("utf-8")

        # Using the specific prompt from the original code
        completion = client.chat.completions.create(
            model=VL_MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "You are an OCR expert specialized in academic question papers. "
                                "Extract all the visible text from this question paper image exactly as it appears. "
                                "Follow these specific guidelines:\n"
                                "1. Start with 'QUESTIONS:'\n"
                                "2. For each question, start with 'QUESTION X:' where X is the question number\n"
                                "3. Include marks in square brackets [X] where present\n"
                                "4. Preserve 'OR' sections exactly as they appear\n"
                                "5. Maintain all mathematical expressions, symbols, and formatting\n"
                                "6. Do not add any explanations or interpretations\n"
                                "7. Do not include any thought process or internal reasoning\n"
                                "8. Preserve exact spacing and line breaks between questions\n"
                                "9. Extract any instructions or notes exactly as they appear\n"
                                "10. Maintain sub-points and numbering exactly as shown"
                            )
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}  # Assuming JPEG/PNG common
                        }
                    ]
                }
            ]
        )
        extracted_text = completion.choices[0].message.content

        # Clean up (same as original code) - maybe rename clean_vl_output later if needed
        cleaned_text = (
            extracted_text
            .replace("◁think▷", "")  # Remove any thinking markers if present
            .replace("\n\n\n", "\n\n")  # Reduce excessive line breaks
            .strip()
        )

        # Ensure the text starts with "QUESTIONS:" (same as original)
        if not cleaned_text.startswith("QUESTIONS:"):
            cleaned_text = "QUESTIONS:\n\n" + cleaned_text  # Add prefix if missing

        print("Question paper text extracted successfully.")
        return cleaned_text

    except Exception as e:
        print(f"ERROR extracting text from question paper '{os.path.basename(image_path)}': {e}")
        print(traceback.format_exc())
        return None

def extract_text_from_images(image_paths: List[str], context_question_paper_text: str, client: OpenAI, VL_MODEL_NAME: str) -> str | None:
    """
    Use VL model to extract text from a list of answer sheet images,
    using the question paper text as context.
    Returns structured answer text or None on error.
    """
    if not client:
        print("ERROR: OpenRouter client not initialized. Cannot call API.")
        return None
    if not image_paths:
        print("ERROR: No image paths provided for text extraction.")
        return None
    if not context_question_paper_text:
        print("WARNING: Question paper text context is missing. Extraction quality may be reduced.")
        context_question_paper_text = "QUESTIONS:\n[Context not available]"  # Provide placeholder

    print(f"Extracting text from {len(image_paths)} answer image(s) using {VL_MODEL_NAME}...")

    # Derive valid question numbers from the context (QP text)
    valid_questions = set()
    # Improved regex to handle various Q formats like "QUESTION 1:", "Q. 2", "Q3)" etc.
    question_matches = re.finditer(r'(?:QUESTION|Q\.?)\s*(\d+[a-zA-Z]?)\s*[:.)]?', context_question_paper_text,
                                   re.IGNORECASE)
    for match in question_matches:
        q_num_str = match.group(1)
        # Try converting to int, handle sub-questions like '3a' if needed later
        try:
            # For now, just store the string identifier found
            valid_questions.add(q_num_str)
        except ValueError:
            print(f"Warning: Could not parse question identifier '{q_num_str}' as integer.")
            valid_questions.add(q_num_str)  # Keep non-integer identifiers as strings

    print(f"Derived valid question identifiers from QP: {sorted(list(valid_questions))}")
    if not valid_questions:
        print("WARNING: No valid question identifiers found in the question paper text. Segmentation might fail.")
        # Decide how to proceed: maybe attempt extraction without filtering? Or stop?
        # For now, proceed but the VL model might struggle without valid Q numbers.

    # Prepare image contents for the API call
    image_contents = []
    for idx, image_path in enumerate(image_paths):
        if not os.path.exists(image_path):
            print(f"WARNING: Image file not found: {image_path}. Skipping.")
            continue
        try:
            with open(image_path, "rb") as img_file:
                base64_image = base64.b64encode(img_file.read()).decode("utf-8")
            image_contents.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}  # Assuming JPEG/PNG
            })
        except Exception as e:
            print(f"Error reading or encoding image {image_path}: {e}")

    if not image_contents:
        print("ERROR: No valid images could be prepared for API call.")
        return None

    # Create the prompt using the structure from original code
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You are an OCR expert specialized in academic answer papers. "
                        f"You will be analyzing {len(image_contents)} images that may contain parts of the same answers. "
                        "\nEXTRACTION RULES:"
                        "\n1. Valid question numbers/identifiers from question paper: " + ", ".join(
                            sorted(list(valid_questions))) +  # Use derived identifiers
                        f"\n\nReference Question Paper Context:\n{context_question_paper_text}\n"  # Provide QP context
                        "\n2. Multi-Image Processing Rules:"
                        "\n   - Combine content from all images to create complete answers for each question number/identifier."
                        "\n   - Avoid duplicating answers for the same question number/identifier."
                        "\n   - Maintain chronological order of content as it appears across images."
                        "\n   - Ensure seamless integration of content belonging to the same answer, even if split across images."
                        "\n3. Text Extraction Guidelines:"
                        "\n   - Start each unique answer with 'QUESTION X:' (where X is one of the valid numbers/identifiers listed above)."
                        "\n   - Each valid question number/identifier should appear at most once in the final output."
                        "\n   - Maintain consistent formatting for the extracted answer text."
                        "\n   - Preserve all mathematical expressions, code blocks, tables, and symbols exactly as they appear in the answer."
                        "\n4. Critical Rules:"
                        "\n   - ONLY structure output using the valid question numbers/identifiers provided."
                        "\n   - If content cannot be associated with a valid number/identifier, ignore it."
                        "\n   - Ensure the text following 'QUESTION X:' corresponds to the answer for that question."
                        "\n   - Do not add any explanations, interpretations, or summaries."
                        "\n   - Do not include any processing notes or thinking steps (like ◁think▷)."
                        "\n   - Remove any duplicate 'QUESTION X:' tags if they accidentally occur for the same X."
                        "\nProcess all images together and provide a single coherent text output containing all the extracted answers, correctly labelled and combined."
                    )
                }
            ] + image_contents  # Append the list of image objects
        }
    ]

    # Make the API call
    try:
        completion = client.chat.completions.create(
            model=VL_MODEL_NAME,
            messages=messages
            # Consider adding max_tokens if needed, though VL models often handle this well
            # max_tokens=4096 # Example
        )
        extracted_text = completion.choices[0].message.content

        # Clean up the extracted text (same cleaning as original code)
        # Rename to avoid potential future conflicts? -> clean_vl_output
        # New - Renaming this function
        def clean_vl_output(vl_text):
            """Cleans VL model specific output artifacts."""
            cleaned = vl_text
            # Remove artifacts seen in original code
            cleaned = cleaned.replace("◁think▷", "")
            cleaned = cleaned.replace("Valid Question Numbers:", "")  # Remove potential meta-text
            cleaned = cleaned.replace("Question Paper Text:", "")  # Remove potential meta-text
            # Remove other common VL model preamble/postamble if observed
            cleaned = re.sub(r"^\s*Here is the extracted text.*?\n+", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\n+Okay, I have processed the images.*$", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
            cleaned = cleaned.strip()
            # Ensure consistent spacing around QUESTION tags (add newline before if missing)
            cleaned = re.sub(r'(?<!\n)\n?(QUESTION \S+:)', r'\n\n\1', cleaned)
            cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)  # Consolidate newlines
            return cleaned.strip()

        cleaned_text = clean_vl_output(extracted_text)

        print(f"Answer text extracted successfully from {len(image_contents)} image(s).")
        return cleaned_text

    except Exception as e:
        print(f"ERROR during VL API call for answer extraction: {e}")
        print(traceback.format_exc())
        return None
    
def preprocess_text(text):
    """
    Preprocess the raw text extracted by the VL model to remove metadata/notes.
    (This function's logic is now inside `clean_vl_output` called by `extract_text_from_images`)
    """
    # The actual cleaning happens within extract_text_from_images now using clean_vl_output.
    # This function remains mostly for compatibility if it was called elsewhere,
    # but ideally, the output from extract_text_from_images is already cleaned.
    # We can simply return the text as is, assuming it's cleaned by the caller.
    if text is None:
        return None
    # Or, apply the cleaning again just in case:
    return text  # Assuming extract_text_from_images already returns cleaned text.
    
def convert_to_dict(text):
    """
    Convert the structured VL text output ("QUESTION X: ...") into a dictionary.
    Keys are integers, values are the corresponding answer strings.
    Handles potential errors during parsing.
    """
    qa_dict = {}
    if not text:
        print("Warning: Cannot convert empty text to dictionary.")
        return qa_dict

    # Split text based on "QUESTION X:" pattern. Handles variations like "QUESTION 1:", "QUESTION 3a:"
    # Positive lookahead `(?=...)` ensures the delimiter is not consumed.
    # Handles potential whitespace variations.
    questions = re.split(r'(?=QUESTION\s+\S+\s*:)', text.strip(), flags=re.IGNORECASE)
    questions = [q.strip() for q in questions if q.strip()]  # Filter out empty strings

    if not questions:
        print("Warning: No 'QUESTION X:' patterns found in the text to split by.")
        # Maybe return the whole text under a default key?
        # qa_dict[0] = text
        return qa_dict

    for entry in questions:
        # Match "QUESTION X:" at the beginning of the string
        match = re.match(r'QUESTION\s+(\S+)\s*:(.*)', entry, re.IGNORECASE | re.DOTALL)
        if match:
            question_id_str = match.group(1).strip()
            answer = match.group(2).strip()
            try:
                # Convert key to integer - THIS IS CRUCIAL for matching marks/key dicts
                question_id_int = int(question_id_str)
                qa_dict[question_id_int] = answer
            except ValueError:
                print(
                    f"Warning: Could not convert question identifier '{question_id_str}' to integer. Storing as string key.")
                # Decide how to handle non-integer keys if they shouldn't occur.
                # Option 1: Store as string (might break later matching)
                qa_dict[question_id_str] = answer
                # Option 2: Skip this entry
                # print(f"Skipping entry with non-integer identifier: {question_id_str}")
                # continue
        else:
            print(f"Warning: Could not parse entry starting with: '{entry[:50]}...'")

    # Add a check for duplicate keys (if the VL model accidentally repeats a QUESTION tag)
    # This implementation implicitly takes the *last* occurrence found due to dict assignment.

    if not qa_dict:
        print("Warning: Resulting dictionary is empty after parsing.")

    return qa_dict

def process_answer(student_answer, reference_answer, result):
    """
    Process student answer against reference answer and calculate metrics.
    Returns updated result object with details and final score.
    """
    try:
        # Apply appropriate weights based on answer type
        weights = get_weights_by_answer_type(result.answer_type)
        
        # Calculate individual metrics
        result.details['semantic_factual_similarity'] = get_semantic_factual_similarity(student_answer, reference_answer)
        result.details['length_appropriateness'] = check_length_ratio(student_answer, reference_answer)
        result.details['key_phrases'] = check_key_phrases(student_answer, reference_answer)
        result.details['coherence'] = 1.0  # Default value; replace with actual coherence metric if implemented
        
        # Calculate sequence alignment if relevant for this answer type
        if 'sequence_alignment' in weights:
            result.details['sequence_alignment'] = check_sequence_alignment(student_answer, reference_answer)
        
        # Type-specific metrics
        if result.answer_type == "code":
            # Placeholder for code structure analysis
            result.details['code_structure'] = 0.8  # Placeholder value
        elif result.answer_type == "mathematical":
            # Placeholder for math expression analysis
            result.details['mathematical_expressions'] = 0.8  # Placeholder value
        elif result.answer_type == "table":
            # Placeholder for table structure analysis
            result.details['table_structure'] = 0.8  # Placeholder value
        
        # Calculate final score as weighted sum of metrics
        final_score = 0.0
        for metric, score in result.details.items():
            if metric in weights:
                final_score += score * weights[metric]
        
        result.final_score = max(0.0, min(1.0, final_score))  # Ensure score is in [0, 1]
        
    except Exception as e:
        result.error = str(e)
        result.details['error'] = str(e)
        result.final_score = 0.0  # Zero score on error
    
    return result


def process_answer_sheet_file(answer_sheet_path, question_paper_text_context, image_output_dir, client, VL_MODEL_NAME):
    """
    Handles PDF/Image input for answer sheet, performs OCR/Structuring via VL,
    and returns the structured dictionary.
    """
    image_paths = []
    if isinstance(answer_sheet_path, str) and answer_sheet_path.lower().endswith(".pdf"):
        image_paths = pdf_to_images(answer_sheet_path, output_folder=image_output_dir)
    elif isinstance(answer_sheet_path, str) and os.path.isfile(answer_sheet_path):  # Single image file
        image_paths = [answer_sheet_path]
    elif isinstance(answer_sheet_path, list):  # List of image files
        image_paths = answer_sheet_path
    else:
        print(
            f"ERROR: Invalid answer_sheet_path format: {answer_sheet_path}. Provide PDF path, image path, or list of image paths.")
        return None

    if not image_paths:
        print("ERROR: No images found or generated for the answer sheet.")
        return None

    # Extract text using VL model
    extracted_answer_text = extract_text_from_images(image_paths, question_paper_text_context, client, VL_MODEL_NAME)

    if not extracted_answer_text:
        print("ERROR: Failed to extract text from answer sheet images.")
        return None

    # Convert structured text to dictionary (ensure integer keys)
    student_answers_dict = convert_to_dict(extracted_answer_text)

    return student_answers_dict
    
def process_key_file_to_dict(answer_key_path, question_paper_text_context, image_output_dir, client, VL_MODEL_NAME):
    """
    Handles PDF/Image input for answer key, performs OCR/Structuring via VL,
    and returns the structured dictionary {int: answer_string}.
    (Very similar to process_answer_sheet_file)
    """
    image_paths = []
    if isinstance(answer_key_path, str) and answer_key_path.lower().endswith(".pdf"):
        image_paths = pdf_to_images(answer_key_path, output_folder=image_output_dir)
    elif isinstance(answer_key_path, str) and os.path.isfile(answer_key_path):  # Single image file
        image_paths = [answer_key_path]
    elif isinstance(answer_key_path, list):  # List of image files
        image_paths = answer_key_path
    else:
        print(
            f"ERROR: Invalid answer_key_path format: {answer_key_path}. Provide PDF path, image path, or list of image paths.")
        return None

    if not image_paths:
        print("ERROR: No images found or generated for the answer key.")
        return None

    # Extract text using VL model - Use the same function, prompt should be general enough
    # Might consider a slightly different prompt if keys need different handling, but unlikely.
    print("\n--- Processing Answer Key File ---")
    extracted_key_text = extract_text_from_images(image_paths, question_paper_text_context, client, VL_MODEL_NAME)

    if not extracted_key_text:
        print("ERROR: Failed to extract text from answer key file.")
        return None

    # Convert structured text to dictionary (ensure integer keys)
    reference_answers_dict = convert_to_dict(extracted_key_text)
    print("Finished processing answer key file.")

    return reference_answers_dict
    
def detect_answer_type(reference_text):
    """Detect the type of answer (code, mathematical, table, text) to adjust scoring weights."""
    if not isinstance(reference_text, str):
        return "text"  # Handle non-string input

    # Compile regex patterns once (moved outside for efficiency, though function scope is ok)
    code_pattern = re.compile(r'\b(void|int|char|float|double|struct|class|def|import|while|for|if|else|return)\b',
                              re.IGNORECASE)
    # More robust math pattern - includes common functions, greek letters (approximation), operators
    math_pattern = re.compile(
        r'(\$\$.*?\$\$|\\[a-zA-Z]+|\b(sin|cos|tan|log|exp|sqrt)\b|[+\-*/=<>≤≥^]|[ΣΠ∫]|[α-ωΑ-Ω])')
    table_pattern = re.compile(r'[|\+][-+]{3,}[|\+]')  # Requires at least 3 dashes for a line

    # Check for code keywords (increase threshold slightly?)
    if len(code_pattern.findall(reference_text)) > 4:  # Original was > 3
        return "code"

    # Check for mathematical content (increase threshold slightly?)
    if len(math_pattern.findall(reference_text)) > 6:  # Original was > 5
        return "mathematical"

    # Check for tables (more robust check)
    if table_pattern.search(reference_text) or reference_text.count('\n|') > 1:  # Look for multiple lines starting with |
        return "table"

    # Default
    return "text"

# --- Grading Weights ---
# Define weights as constants (from original code)
WEIGHTS = {
    "code": {
        'semantic_factual_similarity': 0.40,
        'code_structure': 0.30,  # Specific metric for code
        'key_phrases': 0.10,
        'length_appropriateness': 0.10,
        'coherence': 0.05,
        'sequence_alignment': 0.05
    },
    "mathematical": {
        'semantic_factual_similarity': 0.50,
        'mathematical_expressions': 0.30,  # Specific metric for math
        'key_phrases': 0.10,
        'length_appropriateness': 0.05,
        'coherence': 0.05
        # Sequence alignment might be less relevant for math? Original didn't include it.
    },
    "table": {
        'semantic_factual_similarity': 0.40,
        'table_structure': 0.30,  # Specific metric for tables
        'key_phrases': 0.15,
        'length_appropriateness': 0.10,
        'coherence': 0.05
    },
    "text": {
        'semantic_factual_similarity': 0.65,  # Higher weight for general text
        'length_appropriateness': 0.15,
        'key_phrases': 0.10,
        'coherence': 0.05,
        'sequence_alignment': 0.05
    },
    # New: Handle error case gracefully
    "error": {
        'semantic_factual_similarity': 0.0,
        'length_appropriateness': 0.0,
        'key_phrases': 0.0,
        'coherence': 0.0,
        'sequence_alignment': 0.0
    }
}

def get_weights_by_answer_type(answer_type):
    """Return appropriate weights based on answer type."""
    return WEIGHTS.get(answer_type, WEIGHTS["text"])  # Default to text weights if type unknown

def calculate_bert_score(texts1, texts2, lang='en'):
    """Calculates BERTScore (P, R, F1) safely."""
    try:
        # Ensure inputs are lists of strings
        if isinstance(texts1, str):
            texts1 = [texts1]
        if isinstance(texts2, str):
            texts2 = [texts2]
        if not texts1 or not texts2 or not texts1[0] or not texts2[0]:
            # Handle empty input cases
            return 0.0, 0.0, 0.0

        # device = 'cuda' if torch.cuda.is_available() else 'cpu' # Optional: specify device
        P, R, F1 = bert_score.score(texts1, texts2, lang=lang, verbose=False,
                                    model_type='bert-base-uncased')  # Specify model for consistency?
        # Return mean scores if multiple pairs were processed, or single score otherwise
        return P.mean().item(), R.mean().item(), F1.mean().item()
    except Exception as e:
        print(f"Error calculating BERTScore: {e}")
        # print(f"Texts involved: {texts1}, {texts2}") # Debugging
        return 0.0, 0.0, 0.0  # Return zero scores on error
    
def get_semantic_factual_similarity(student_text, reference_text):
    """Calculate combined semantic (BERTScore F1) and factual (number match) similarity."""
    if not isinstance(student_text, str) or not isinstance(reference_text, str):
        return 0.0

    # Semantic Similarity using BERTScore F1
    _, _, semantic_similarity_f1 = calculate_bert_score(student_text, reference_text)

    # Factual Accuracy based on numbers
    number_pattern = re.compile(r'\b\d+(?:\.\d+)?\b')  # Match standalone numbers
    student_numbers = set(number_pattern.findall(student_text))
    reference_numbers = set(number_pattern.findall(reference_text))

    # Calculate intersection over union for numbers? Or just intersection / reference?
    # Original used intersection / reference. Stick to that unless specified otherwise.
    if not reference_numbers:
        factual_accuracy = 1.0 if not student_numbers else 0.5  # Both empty = 1.0; Student has extra = lower score?
    else:
        intersection_count = len(student_numbers.intersection(reference_numbers))
        factual_accuracy = intersection_count / len(reference_numbers)
        # Penalize slightly if student introduces numbers not in reference? Optional.
        # extra_numbers = len(student_numbers - reference_numbers)
        # penalty = extra_numbers * 0.1
        # factual_accuracy = max(0, factual_accuracy - penalty)

    # Combine scores (original weights: 70% semantic, 30% factual)
    combined_score = (semantic_similarity_f1 * 0.7 + factual_accuracy * 0.3)
    return max(0.0, min(1.0, combined_score))  # Ensure score is within [0, 1]
    
def check_length_ratio(student_text, reference_text):
    """Check if student answer length is appropriate relative to reference length (word count)."""
    if not isinstance(student_text, str) or not isinstance(reference_text, str):
        return 0.0

    try:
        student_length = len(word_tokenize(student_text))
        reference_length = len(word_tokenize(reference_text))
    except Exception as e:
        print(f"Error tokenizing for length ratio: {e}")
        return 0.0  # Return 0 if tokenization fails

    if reference_length == 0:
        return 0.0 if student_length > 0 else 1.0  # Perfect match if both empty, else 0

    ratio = student_length / reference_length

    # Original logic: score decreases if too long (beyond 1.5x) or too short.
    if ratio <= 1.5 and ratio >= 0.5:  # Allow some flexibility (e.g., within 50%-150%)
        # Score proportional to ratio, capped at 1.0
        # Map ratio [0.5, 1.5] to score [0.5, 1.0] (roughly) - simpler approach below
        return min(1.0, ratio) if ratio <= 1.0 else (1.0 - (ratio - 1.0) * 0.5)  # Penalize length > 1.0
    elif ratio < 0.5:
        return ratio * 2  # Linear penalty for too short (0.5 ratio -> 1.0 score seems wrong, maybe just ratio?) Let's use ratio directly. --> return ratio
        # Corrected approach: return ratio if too short.
        return max(0.0, ratio)  # Score is the ratio itself if too short
    else:  # ratio > 1.5
        # Penalize excess length more sharply
        return max(0.0, 1.0 - (ratio - 1.5) * 0.5)  # Example penalty: score drops from 1 at 1.5x

    # Simpler original logic interpretation:
    # if ratio <= 1.5: return min(1.0, ratio) # Score = ratio, capped at 1 if student is shorter
    # else: return min(1.0, 1.5 / ratio) # Score decreases inverse proportionally if student is longer

    # Let's stick to the *second interpretation* of the original logic as it's simpler:
    if reference_length == 0:
        return 1.0 if student_length == 0 else 0.0
    ratio = student_length / reference_length
    if ratio <= 1.0:  # Student answer is shorter or equal
        # Score is proportional to length, up to reference length
        return ratio
    elif ratio <= 1.5:  # Student answer is slightly longer (up to 1.5x) - Full score? Or slight penalty?
        # Original: min(1.0, 1.5/ratio) -> gives 1.0 at ratio=1.5. Let's use this.
        return 1.0  # Allow up to 1.5x length without penalty? Let's try this.
        # Original logic: return min(1.0, 1.5 / ratio) # Gives 1.0 at ratio 1.5, <1 above that.
    else:  # ratio > 1.5
        # Score decreases
        return max(0.0, min(1.0, 1.5 / ratio))  # Original logic was likely this
    
def check_sequence_alignment(student_text, reference_text):
    """Check structural alignment (sentence and paragraph counts)."""
    if not isinstance(student_text, str) or not isinstance(reference_text, str):
        return 0.0

    try:
        student_sentences = sent_tokenize(student_text)
        reference_sentences = sent_tokenize(reference_text)
        # Handle potential tokenization issues leading to empty lists
        if not student_sentences and not reference_sentences:
            return 1.0
        if not student_sentences or not reference_sentences:
            return 0.0

        # Avoid division by zero - use max length as denominator
        max_sentences = max(len(student_sentences), len(reference_sentences))
        sent_ratio = min(len(student_sentences), len(reference_sentences)) / max_sentences if max_sentences > 0 else 1.0

        student_paragraphs = [p for p in student_text.split('\n\n') if p.strip()]
        reference_paragraphs = [p for p in reference_text.split('\n\n') if p.strip()]
        if not student_paragraphs and not reference_paragraphs:
            return 1.0
        if not student_paragraphs or not reference_paragraphs:
            return 0.0

        max_paragraphs = max(len(student_paragraphs), len(reference_paragraphs))
        para_ratio = min(len(student_paragraphs), len(reference_paragraphs)) / max_paragraphs if max_paragraphs > 0 else 1.0

        # Combine sentence and paragraph alignment scores (average)
        return (sent_ratio + para_ratio) / 2
    except Exception as e:
        print(f"Error during sequence alignment: {e}")
        return 0.0
    
def get_phrases(text, max_phrase_length=3):
    """Extract n-grams (phrases) up to max_phrase_length efficiently."""
    if not isinstance(text, str):
        return set()
    try:
        words = word_tokenize(text.lower())
    except Exception as e:
        print(f"Error tokenizing for phrases: {e}")
        return set()

    phrases = set()
    # Generate n-grams from 1 up to max_phrase_length
    for n in range(1, max_phrase_length + 1):
        for i in range(len(words) - n + 1):
            phrases.add(" ".join(words[i:i + n]))
    return phrases
    
def check_key_phrases(student_text, reference_text):
    """Check for presence of key phrases (including synonyms for single words)."""
    if not isinstance(student_text, str) or not isinstance(reference_text, str):
        return 0.0

    reference_phrases = get_phrases(reference_text)
    student_phrases = get_phrases(student_text)

    correct_phrase_count = 0
    for ref_phrase in reference_phrases:
        if ref_phrase in student_phrases:
            correct_phrase_count += 1
        else:
            ref_words = ref_phrase.split()
            # Check for word-level synonyms
            if all(any(similar(word, syn) for syn in get_synonyms(word)) for word in ref_words):
                correct_phrase_count += 1  # Consider valid if all words in phrase have synonyms
    # Original: correct_phrase_count / len(reference_phrases) (issue if reference is empty)
    if not reference_phrases:
        return 1.0 if not student_phrases else 0.5  # Perfect if both empty, otherwise lower

    return correct_phrase_count / len(reference_phrases)

class Result:
    """
    Represents the result of a single question's evaluation.
    """
    def __init__(self, question_id, answer_type):
        self.question_id = question_id
        self.answer_type = answer_type
        self.details = {}  # Detailed breakdown of scores
        self.final_score = 0.0  # Overall score for the question
        self.error = None

def evaluate_student_answers(student_answers: Dict[int, str], reference_answers: Dict[int, str], 
                           rubrics_marks: Dict[int, float], client: OpenAI = client, 
                           vl_model_name: str = VL_MODEL_NAME) -> Dict[str, Any]:
    """
    Evaluate student answers against reference answers.
    """
    total_marks = sum(rubrics_marks.values())
    overall_score = 0.0
    question_results = {}  # Store results per question

    for qid, student_answer in student_answers.items():
        reference_answer = reference_answers.get(qid)
        max_marks = rubrics_marks.get(qid, 0.0)
        if not reference_answer:
            print(f"WARNING: No reference answer for question {qid}. Skipping.")
            continue

        result = Result(qid, detect_answer_type(reference_answer))
        question_results[qid] = result
        
        result = process_answer(student_answer, reference_answer, result)
        result.final_score = result.final_score * max_marks
        overall_score += result.final_score
    
    # Convert Result objects to dictionaries for JSON serialization
    question_results_dict = {}
    for qid, result in question_results.items():
        question_results_dict[qid] = {
            'answer_type': result.answer_type,
            'details': result.details,
            'final_score': result.final_score,
        }
        if result.error:
            question_results_dict[qid]['error'] = result.error
    
    # Calculate percentage
    percentage = (overall_score / total_marks * 100) if total_marks > 0 else 0.0
    percentage = round(percentage, 2)  # Round to 2 decimal places
    
    return {
        'overall_score': overall_score,
        'total_marks': total_marks,
        'percentage': percentage,
        'question_results': question_results_dict
    }