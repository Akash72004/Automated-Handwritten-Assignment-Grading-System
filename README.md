# ABSTRACT

Manual marking of handwritten assignments continues to be a nagging problem in education, prone to
bias, inconsistency, and inefficiency. This study introduces an Automated Handwritten Assignment
Grading System that combines Optical Character Recognition and Natural Language Processing to
offer objective evaluation of subjective answers. The system takes three inputs: student answer sheets,
teacher model answers, and question papers with corresponding rubrics. Applying Llama-4 for text
extraction and OCR, the system breaks responses into segments by question numbers and applies
answer type detection (code, mathematical, tabular, or theoretical). Semantic evaluation uses SBERT
with several assessment parameters: semantic-factual similarity, code structure, key phrases, length
appropriateness, coherence, and sequence alignment. Question-specific dynamic weightings are used
to calculate final scores. Testing conducted on 13 Data Structures and Algorithms answer sheets
demonstrated an accuracy of 87.4% in agreement with expert grading. The system, implemented as a
Flask application, reduces instructor workload considerably while delivering instant, consistent
feedback. The method is an efficient solution for standardizing evaluation across large numbers of
students without compromising fairness or enabling personalized learning through targeted feedback.

# SYSTEM ARCHITECTURE

The Proposed architecture employs a sophisticated multi-stage pipeline designed for accuracy and
flexibility across different answer types. At its core, the system uses a preprocessing stage that
converts all inputs to image format before applying LLaMA-4's advanced OCR capabilities for
structure-preserving text extraction. The question mapping module intelligently aligns student and
teacher answers, accommodating even optional question scenarios. The system's strength lies in its
specialized approach to different answer formats (code, mathematical, tabular, and textual), applying
dynamically weighted evaluation parameters for each type. The evaluation engine combines semantic
similarity measurements with structural analysis using BERT-based technologies, ensuring that both
content accuracy and presentation are considered. Finally, a modular deployment architecture with
separate backend (Python/Flask) and frontend components allows for scalable implementation that
can be readily integrated into existing educational workflows.

![Image](https://github.com/user-attachments/assets/c4e80adc-53bf-470c-b006-c90ee7d86b92)

The below figure demonstrates the detailed segmentation logic of how the question paper and the answersheets
are preprocessed to send through the evaluation engine.
![Image](https://github.com/user-attachments/assets/0b4cdc0f-4500-4432-a659-972f5879d521)

The Proposed Evaluation engine as shown below employs a sophisticated multi-stage pipeline
designed for accuracy and flexibility across different answer types. 
![Image](https://github.com/user-attachments/assets/4c47544f-f94d-4bcb-ab79-a55a12ac651a)

# RESULT ANALYSIS

The automated grading system was rigorously tested on actual student assignments to evaluate its
performance and identify areas for improvement. Below table presents the comparison between system
predicted scores and actual scores for 13 students, along with the calculated accuracy for each student.

Evaluation results for students
![Image](https://github.com/user-attachments/assets/0babc934-3c2c-4f23-9596-1e7946f9db9c)

![Image](https://github.com/user-attachments/assets/34802c39-cc65-414b-a134-628e35b9c763)

The overall system accuracy was calculated using the Mean Absolute Error (MAE) and converting it
to a percentage accuracy relative to the maximum possible score of 15 marks. The MAE was
calculated to be 1.87, resulting in an average system accuracy of 87.46%.
Detailed analysis of individual student evaluations revealed varying levels of alignment with human
grading. For students like Jeet, Agnya, Ayush, and Disha, the system achieved high accuracy rates
25
above 95%, indicating strong alignment with human judgment. However, more significant
discrepancies were observed for students like Harshil and Kashvi, where the system substantially
underestimated performance with accuracies of 63.33% and 73.33% respectively.
For instance, Ayush received a system score of 9 out of 15, compared to an actual score of 9.5. In
Question 1, the final scaled score was 1.64, with a semantic similarity of 0.627 and length
appropriateness of 0.33, indicating a strong semantic alignment but a need for more detail. Question
2 had a final score of 5.03, with a semantic similarity of 0.678 and length appropriateness of 0.997,
reflecting a well-aligned and detailed response. Question 3 showed a final score of 2.72, with a
semantic similarity of 0.239 and length appropriateness of 0.964, suggesting that while the structure
was correct, the content needed enhancement.
Kashvi’s system score was 8 out of 15, while the actual score was 12. In Question 1, the final scaled
score was 1.17, with a semantic similarity of 0.351 and length appropriateness of 0.469, indicating
moderate alignment and detail. Question 2 had a final score of 3.61, with a semantic similarity of
0.424 and length appropriateness of 0.843, reflecting a well-structured response. Question 3 showed
a final score of 2.72, with a semantic similarity of 0.6 and length appropriateness of 1.0, indicating a
strong alignment and appropriate detail.
![Image](https://github.com/user-attachments/assets/50271845-1d11-49bc-8e28-be948e244f12)
ABove figure presents a visual comparison of predicted versus actual marks across all students. The
visualization reveals that the system demonstrated a tendency to underestimate scores for
highperforming students (those scoring above 12 marks), while showing reasonable accuracy for
moderate performers. This pattern suggests that the system may not adequately capture exceptional
responses that demonstrate deeper understanding or creativity beyond the expected answer patterns. 

# CONCLUSION

This study presents a comprehensive framework for automated assignment grading, combining
multi-metric analysis, dynamic parameter weighting, and real-world validation. The system
achieved reasonable accuracy in assessing structured responses, but discrepancies with human
grading underscored the challenges in evaluating nuanced answers. The integration of BERTScore
for semantic analysis and structural metrics for content evaluation provided a balanced approach,
though further refinements are necessary to improve consistency.
The implemented approach of classifying response types before applying specialized weight
configurations represents a significant advancement in automated grading technology. By
recognizing the unique requirements of code-based, mathematical, tabular, and text responses, the
system demonstrated adaptability across diverse assessment scenarios. This adaptive strategy
contributed to the overall accuracy improvement from 72% in early testing to 87.46% in the final
evaluation.
Future improvements should focus on enhancing semantic models to better capture conceptual
variations, optimizing parameter weighting to reduce biases such as over-reliance on response length,
and incorporating human oversight for borderline cases. Particular attention should be given to
improving the evaluation of high-performing responses that demonstrate exceptional understanding
or creative approaches, as these were consistently undervalued in the current implementation.
Additionally, continued refinement of the type-specific weighting systems could further improve
accuracy across different question formats. For code-based responses, enhanced syntax and structure
analysis could better capture the quality of implementation. For mathematical responses, improved
recognition of equivalent expressions and solution paths could address current limitations in
evaluating creative problem-solving approaches.
By addressing these limitations, the system can evolve into a more reliable tool for automated grading,
offering scalability and fairness in educational assessment. The findings highlight the potential of
hybrid approaches that combine advanced NLP techniques with adaptive weighting strategies to
bridge the gap between automated and human evaluation. With continued development,such systems
could provide valuable support to educators, reducing grading workload while maintaining
assessment quality, particularly in large-scale educational environments.

# FUTURE SCOPE

The future of automated grading and text processing systems lies in developing more sophisticated
and educationally meaningful solutions. A key direction involves creating context-aware AI models
capable of evaluating not just factual accuracy but also the quality of arguments, logical flow, and
conceptual understanding in student responses. These systems should seamlessly integrate
multimodal capabilities to handle diverse content formats including text, mathematical equations,
diagrams, and handwritten elements, mirroring real-world educational materials. To enhance their
pedagogical value, future systems must generate personalized, actionable feedback that helps
students improve, rather than simply providing numerical scores. Efficiency improvements through
techniques like model optimization will be crucial for making these solutionsscalable across different
educational settings. Equally important is the development of comprehensive, representative datasets
that capture the full spectrum of student responses across subjects and languages. Ultimately, the
most effective solutions will likely emerge from thoughtful human-AI collaboration frameworks that
combine the consistency of automated systems with the nuanced understanding of human educators,
ensuring both fairness and educational effectiveness in assessment practices.
