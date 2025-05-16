from flask import Flask, request, render_template, send_from_directory
import os
from werkzeug.utils import secure_filename
from openai import OpenAI
from flask_cors import CORS



# Import all helper functions
from helper_functions import (
    extract_text_from_question_paper,
    process_answer_sheet_file,
    process_key_file_to_dict,
    evaluate_student_answers,
    Result
)

# Initialize OpenAI client
OPENROUTER_API_KEY = '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'
VL_MODEL_NAME = "meta-llama/llama-4-maverick:free"
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)

app = Flask(__name__)  # Corrected double underscores
CORS(app)

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'  # Where uploaded files will be stored
IMAGE_OUTPUT_FOLDER = "temp_images_output"  # For images generated from PDF

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER  # Configure the upload folder
app.config['IMAGE_OUTPUT_FOLDER'] = IMAGE_OUTPUT_FOLDER

ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg'}  # Allowed file extensions

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

Final Postfix Expression: ABCD-E^FGHI*+-/+
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

# --- Helper function to check allowed file extensions ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Routes ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/grade', methods=['POST'])
def grade():
    if request.method == 'POST':
        # File Handling
        question_paper_file = request.files['question_paper']
        answer_sheet_file = request.files['answer_sheet']
        answer_key_file = request.files['answer_key']

        if not question_paper_file or not answer_sheet_file or not answer_key_file:
            return {"error": "Please provide all the files"}, 400
        
        # Save the files to the upload folder
        question_paper_filename = secure_filename(question_paper_file.filename)
        question_paper_path = os.path.join(app.config['UPLOAD_FOLDER'], question_paper_filename)
        question_paper_file.save(question_paper_path)
        
        answer_sheet_filename = secure_filename(answer_sheet_file.filename)
        answer_sheet_path = os.path.join(app.config['UPLOAD_FOLDER'], answer_sheet_filename)
        answer_sheet_file.save(answer_sheet_path)
        
        answer_key_filename = secure_filename(answer_key_file.filename)
        answer_key_path = os.path.join(app.config['UPLOAD_FOLDER'], answer_key_filename)
        answer_key_file.save(answer_key_path)
        
        # Start Grading Process
        try:
            question_paper_text = extract_text_from_question_paper(question_paper_path, client, VL_MODEL_NAME)
            if not question_paper_text:
                return {"error": "Failed to process question paper. Cannot proceed."}, 500
            
            student_answers_dict = process_answer_sheet_file(
                answer_sheet_path,
                question_paper_text,
                app.config['IMAGE_OUTPUT_FOLDER'],
                client,
                VL_MODEL_NAME
            )
            
            if not student_answers_dict:
                return {"error": "Failed to process student answer sheet. Cannot proceed."}, 500
            
            ANSWER_KEY_SOURCE = 'dict' if not answer_key_path else 'file'
            final_reference_answers_dict = None

            if ANSWER_KEY_SOURCE == 'file':
                final_reference_answers_dict = process_key_file_to_dict(
                    answer_key_path,
                    question_paper_text,
                    app.config['IMAGE_OUTPUT_FOLDER'], 
                    client, 
                    VL_MODEL_NAME
                )
            elif ANSWER_KEY_SOURCE == 'dict':
                try:
                    final_reference_answers_dict = {int(k): v for k, v in ANSWER_KEY_DICT.items()}
                except ValueError:
                    return {"error": "Keys in predefined ANSWER_KEY_DICT must be integers."}, 500
            
            if not final_reference_answers_dict:
                return {"error": "Failed to load or process answer key. Cannot proceed."}, 500
            
            try:
                final_marks_dict = {int(k): v for k, v in MARKS_PER_QUESTION.items()}
            except ValueError:
                return {"error": "Keys in MARKS_PER_QUESTION must be integers."}, 500
            
            evaluation_results = evaluate_student_answers(
                student_answers=student_answers_dict,
                reference_answers=final_reference_answers_dict,
                rubrics_marks=final_marks_dict,
                client=client,
                vl_model_name=VL_MODEL_NAME
            )

            if evaluation_results:
                # Return the results as JSON
                return {
                    "overall_score": evaluation_results['overall_score'],
                    "total_marks": evaluation_results['total_marks'],
                    "percentage": evaluation_results['percentage'],
                    "question_results": evaluation_results['question_results']
                }, 200
            else:
                return {"error": "Evaluation could not be completed due to errors."}, 500
        except Exception as e:
            return {"error": f"An error occurred: {str(e)}"}, 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':  # Corrected double underscores
    # Create directories if they don't exist
    if not os.path.exists(UPLOAD_FOLDER): 
        os.makedirs(UPLOAD_FOLDER)
    if not os.path.exists(IMAGE_OUTPUT_FOLDER): 
        os.makedirs(IMAGE_OUTPUT_FOLDER)
    app.run(debug=True)