# **App Name**: GradeWise

## Core Features:

- Document Upload: Securely upload student answer sheets (PDF or images), question papers (images), and answer keys (PDF, images, or direct input).
- AI-Powered Grading: Leverage generative AI to evaluate student answers based on semantic similarity, coherence, and other relevant metrics, providing detailed marks breakage for each question. The LLM will use the question paper, answer key, and rubrics as tools to accurately evaluate the student's answer.
- Results Dashboard: Display the student's overall score and detailed marks breakage per question, including the AI's analysis of semantic similarity and coherence.

## Style Guidelines:

- Primary color: Clean white (#FFFFFF) for a professional and uncluttered look.
- Secondary color: Light gray (#F0F0F0) for backgrounds and subtle dividers.
- Accent: Teal (#008080) to indicate actionable components.
- Clear and readable font for all text elements.
- Use a consistent set of icons for file types (PDF, image) and actions (upload, grade, view results).
- A clean, well-organized layout with clear sections for uploading documents, viewing results, and error messages.
- Subtle transitions and loading animations to provide feedback during the grading process.

## Original User Request:
this is the final model 
im just pasting code cell 2 and code cell 6 cuz the notebook is too big 
however 
code cell 2 consists of the input files 
which is the student's pdf , answersheet , question paper, predefined answerkey dictionary 

Deploy this on a app 
the UI app should be more like TurnitIn 
make the UI really really GOOD 
where u upload the pdf of the student, question paper/ rubrics, correct answers (or can even set predefined correct answers)  
and in the output you get the marks of the student 
along with marks breakage of each question with the scores (semantic similarity, cohenerance and bla bla)

# Code Cell 2: Configuration - Input Files and Grading Parameters
# ---------------------------------------------------------------

# --- Input File Paths ---
# Path to the question paper (must be an image file like JPEG, PNG)
QUESTION_PAPER_PATH = "/content/question_paper.jpeg" # Replace with your actual path

# Path to the student's answer sheet (can be PDF or an image file/list of image files)
ANSWER_SHEET_PATH = "/content/Joel-D041.pdf" # Replace with your actual path (PDF or image)
# Example for multiple images:
# ANSWER_SHEET_PATH = ["page1.jpg", "page2.jpg"]

# Path to the answer key (optional, can be PDF, image, list of images, or 'dict')
ANSWER_KEY_PATH = "/content/Spam_AnswerKey.pdf" # Replace with actual path or set ANSWER_KEY_SOURCE to 'dict'
# Example for multiple images:
# ANSWER_KEY_PATH = ["key_page1.jpg", "key_page2.jpg"]

# Define how the Answer Key is provided:
# 'file': Process ANSWER_KEY_PATH using VL model OCR.
# 'dict': Use the predefined dictionary ANSWER_KEY_DICT below.
ANSWER_KEY_SOURCE = 'file' # Change to 'file' if using a file

# --- Predefined Answer Key (if ANSWER_KEY_SOURCE = 'dict') ---
# Keys MUST be integers corresponding to question numbers.
ANSWER_KEY_DICT = {
    1: """
Asymptotic notations are mathematical tools used to describe the running time or space complexity of algorithms:

1. Big-O Notation (O):
   - Represents the upper bound/worst-case complexity
   - f(n) = O(g(n)) means f(n) grows no faster than g(n)
   - Example: Bubble Sort has O(n²) in worst case

2. Omega Notation (Ω):
   - Represents the lower bound/best-case complexity
   - f(n) = Ω(g(n)) means f(n) grows at least as fast as g(n)
   - Example: Bubble Sort has Ω(n) when array is already sorted

3. Theta Notation (Θ):
   - Represents both upper and lower bounds (tight bound)
   - f(n) = Θ(g(n)) means f(n) grows at the same rate as g(n)
   - Example: Matrix multiplication has Θ(n³)

Key Points:
- Big-O is most commonly used for worst-case analysis
- Ω is useful for understanding best-case scenarios
- Θ gives precise average-case analysis for balanced algorithms
""",
    2: """
Operations on Circular Linked List:

1. Insertion at Start:
   newNode = createNode(data)
   if head == NULL:
       head = newNode
       newNode.next = head
   else:
       temp = head
       while temp.next != head:
           temp = temp.next
       temp.next = newNode
       newNode.next = head
       head = newNode

2. Deletion of Last Node:
   if head == NULL: return
   if head.next == head:
       free(head)
       head = NULL
   else:
       current = head
       while current.next.next != head:
           current = current.next
       free(current.next)
       current.next = head

Example:
Initial: 1 → 2 → 3 → (back to 1)
After inserting 0 at start: 0 → 1 → 2 → 3 → (back to 0)
After deleting last node: 0 → 1 → 2 → (back to 0)
""",
    3: """
Infix to Postfix Conversion Table for: A + B * (C - D) ^ E / (F - G + H * I)

| Symbol | Stack (bottom to top) | Output | Action Taken |
|--------|-----------------------|--------|--------------|
| A      | []                    | A      | Output operand |
| +      | [+]                   | A      | Push operator |
| B      | [+]                   | AB     | Output operand |
| *      | [+, *]                | AB     | Push operator (higher precedence than +) |
| (      | [+, *, (]             | AB     | Push parenthesis |
| C      | [+, *, (]             | ABC    | Output operand |
| -      | [+, *, (, -]          | ABC    | Push operator |
| D      | [+, *, (, -]          | ABCD   | Output operand |
| )      | [+, *]                | ABCD-  | Pop until '(' |
| ^      | [+, *, ^]             | ABCD-  | Push operator (highest precedence) |
| E      | [+, *, ^]             | ABCD-E | Output operand |
| /      | [+, *, /]             | ABCD-E^| Pop ^ (higher precedence), then push / |
| (      | [+, *, /, (]          | ABCD-E^| Push parenthesis |
| F      | [+, *, /, (]          | ABCD-E^F | Output operand |
| -      | [+, *, /, (, -]       | ABCD-E^F | Push operator |
| G      | [+, *, /, (, -]       | ABCD-E^FG | Output operand |
| +      | [+, *, /, (, +]       | ABCD-E^FG- | Pop -, push + (same precedence) |
| H      | [+, *, /, (, +]       | ABCD-E^FGH | Output operand |
| *      | [+, *, /, (, +, *]    | ABCD-E^FGH | Push operator (higher precedence than +) |
| I      | [+, *, /, (, +, *]    | ABCD-E^FGHI | Output operand |
| )      | [+, *, /]             | ABCD-E^FGHI*+- | Pop until '(' |
| End    | []                    | ABCD-E^FGHI*+-/ | Pop all remaining operators |

Final Postfix Expression: ABCD-E^*FGHI*+-/+
"""
}

# --- Grading Rubrics/Marks ---
# Keys MUST be integers corresponding to question numbers.
MARKS_PER_QUESTION = {
    1: 3,
    2: 7,
    3: 5,
    # Add max marks for all questions evaluated
}

# --- PDF to Image Conversion Output Folder ---
# New - Made configurable
IMAGE_OUTPUT_FOLDER = "temp_images_output"

# --- Validation ---
# Check if input files exist (basic check)
if not os.path.exists(QUESTION_PAPER_PATH):
    print(f"ERROR: Question paper image not found at: {QUESTION_PAPER_PATH}")
# Add checks for ANSWER_SHEET_PATH and ANSWER_KEY_PATH if ANSWER_KEY_SOURCE=='file'
# ... (add checks as needed)

print("Configuration loaded.")

# Code Cell 6: Main Execution Workflow
# -----------------------------------

print("==============================================")
print("  Automated Assignment Grading System (v2)  ")
print("==============================================")

# --- Preparations ---
# Ensure API client is ready
if not client:
    print("ERROR: OpenRouter API client failed to initialize. Grading cannot proceed.")
    # Exit or raise error
    # exit() # Uncomment to stop execution

# Create temporary directory for images if needed
os.makedirs(IMAGE_OUTPUT_FOLDER, exist_ok=True)

# --- Step 1: Process Question Paper ---
print("\n[Step 1/5] Processing Question Paper...")
question_paper_text = extract_text_from_question_paper(QUESTION_PAPER_PATH)

if not question_paper_text:
    print("ERROR: Failed to process question paper. Cannot proceed.")
    # exit() # Uncomment to stop execution
else:
    # Display snippet of extracted QP text (optional)
    print("Question Paper Text Snippet:")
    print("-" * 30)
    print(question_paper_text[:500] + "...")
    print("-" * 30)

# --- Step 2: Process Student Answer Sheet ---
print("\n[Step 2/5] Processing Student Answer Sheet...")
student_answers_dict = process_answer_sheet_file(
    ANSWER_SHEET_PATH,
    question_paper_text, # Provide QP text as context for VL model
    IMAGE_OUTPUT_FOLDER
)

if not student_answers_dict:
    print("ERROR: Failed to process student answer sheet. Cannot proceed.")
    # exit() # Uncomment to stop execution
else:
    print(f"Successfully processed student answer sheet. Found answers for {len(student_answers_dict)} questions.")
    # Display snippet (optional)
    # print("Sample Student Answer (Q1):", student_answers_dict.get(1, "N/A")[:200] + "...")


# --- Step 3: Load or Process Answer Key ---
print("\n[Step 3/5] Loading/Processing Answer Key...")
final_reference_answers_dict = None

if ANSWER_KEY_SOURCE == 'file':
    final_reference_answers_dict = process_key_file_to_dict(
        ANSWER_KEY_PATH,
        question_paper_text, # Provide QP context here too if needed
        IMAGE_OUTPUT_FOLDER
    )
elif ANSWER_KEY_SOURCE == 'dict':
    # Ensure keys are integers if using predefined dict
    try:
        final_reference_answers_dict = {int(k): v for k, v in ANSWER_KEY_DICT.items()}
        print("Using predefined answer key dictionary.")
    except ValueError:
        print("ERROR: Keys in predefined ANSWER_KEY_DICT must be integers.")
        final_reference_answers_dict = None
else:
    print(f"ERROR: Invalid ANSWER_KEY_SOURCE '{ANSWER_KEY_SOURCE}'. Use 'file' or 'dict'.")

if not final_reference_answers_dict:
    print("ERROR: Failed to load or process answer key. Cannot proceed.")
    # exit() # Uncomment to stop execution
else:
     print(f"Answer key ready. Contains answers for {len(final_reference_answers_dict)} questions.")
     # Display snippet (optional)
     # print("Sample Reference Answer (Q1):", final_reference_answers_dict.get(1, "N/A")[:200] + "...")


# --- Step 4: Prepare Marks Dictionary ---
print("\n[Step 4/5] Preparing Marks Rubric...")
# Ensure keys are integers
try:
    final_marks_dict = {int(k): v for k, v in MARKS_PER_QUESTION.items()}
    print(f"Marks rubric loaded for {len(final_marks_dict)} questions.")
except ValueError:
    print("ERROR: Keys in MARKS_PER_QUESTION must be integers.")
    final_marks_dict = None

if not final_marks_dict:
     print("ERROR: Failed to load marks rubric. Cannot proceed.")
     # exit()

# --- Step 5: Perform Grading ---
print("\n[Step 5/5] Performing Evaluation...")
evaluation_results = None
if student_answers_dict is not None and final_reference_answers_dict is not None and final_marks_dict is not None:
    evaluation_results = evaluate_student_answers(
        student_answers=student_answers_dict,
        reference_answers=final_reference_answers_dict,
        rubrics_marks=final_marks_dict
    )
else:
    print("Skipping evaluation due to errors in previous steps.")


# --- Step 6: Display Results ---
print("\n==============================================")
print("             Grading Results")
print("==============================================")

if evaluation_results:
    print(f"\nOverall Score: {evaluation_results['overall_score']} / {evaluation_results['total_marks']}")
    print(f"Percentage: {evaluation_results['percentage']}%")
    print("\n--- Scores per Question ---")
    for qid, result in sorted(evaluation_results['question_results'].items()):
        marks = final_marks_dict.get(qid, "N/A")
        print(f"\nQuestion {qid} (Max Marks: {marks}, Type: {result.get('answer_type', 'N/A')})")
        print(f"  Score Awarded: {result.get('final_score', 'Error')}")
        print("  Details:")
        if 'details' in result and isinstance(result['details'], dict):
            for metric, score in result['details'].items():
                if metric != 'error': # Don't reprint error here
                     print(f"    - {metric:<30}: {score:.3f}")
                else:
                     print(f"    - Error during evaluation: {score}")
        elif 'final_score' in result and result['final_score'] == 0.0:
             print("    - Evaluation skipped or failed (Score 0).")
        else:
             print("    - No details available.")

else:
    print("Evaluation could not be completed due to errors.")

print("\n==============================================")
print("              End of Report")
print("==============================================")

# Optional: Clean up temporary image folder
# import shutil
# if os.path.exists(IMAGE_OUTPUT_FOLDER):
#     print(f"\nCleaning up temporary image folder: {IMAGE_OUTPUT_FOLDER}")
#     # shutil.rmtree(IMAGE_OUTPUT_FOLDER)
  