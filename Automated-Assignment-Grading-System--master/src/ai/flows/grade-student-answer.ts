'use server';
/**
 * @fileOverview This file defines a Genkit flow for grading student answers using AI.
 *
 * - gradeStudentAnswer - The main function to grade a student's answer sheet.
 * - GradeStudentAnswerInput - The input type for the gradeStudentAnswer function.
 * - GradeStudentAnswerOutput - The output type for the gradeStudentAnswer function.
 */

import {ai} from '@/ai/ai-instance';
import {z} from 'genkit';

const GradeStudentAnswerInputSchema = z.object({
  questionPaperDataUri: z
    .string()
    .describe(
      "A photo of the question paper, as a data URI that must include a MIME type and use Base64 encoding. Expected format: 'data:<mimetype>;base64,<encoded_data>'."
    ),
  answerSheetDataUri: z
    .string()
    .describe(
      "The student's answer sheet, as a data URI that must include a MIME type and use Base64 encoding. Expected format: 'data:<mimetype>;base64,<encoded_data>'."
    ),
  answerKeyDataUri: z
    .string()
    .describe(
      "The answer key, as a data URI that must include a MIME type and use Base64 encoding. Expected format: 'data:<mimetype>;base64,<encoded_data>'."
    ),
  marksPerQuestion: z.record(z.number(), z.number()).describe('A dictionary of question numbers to maximum marks for that question.'),
});

export type GradeStudentAnswerInput = z.infer<typeof GradeStudentAnswerInputSchema>;

const GradeStudentAnswerOutputSchema = z.object({
  overallScore: z.number().describe('The overall score of the student.'),
  totalMarks: z.number().describe('The total possible marks.'),
  percentage: z.number().describe('The percentage score of the student.'),
  questionResults: z.record(
    z.object({
      finalScore: z.number().describe('The score awarded for the question.'),
      details: z.record(z.string(), z.number()).optional().describe('Details of the evaluation metrics.'),
      answer_type: z.string().optional().describe('Type of answer')
    })
  ).
describe('A dictionary of question numbers to evaluation results.'),
});

export type GradeStudentAnswerOutput = z.infer<typeof GradeStudentAnswerOutputSchema>;

export async function gradeStudentAnswer(input: GradeStudentAnswerInput): Promise<GradeStudentAnswerOutput> {
  return gradeStudentAnswerFlow(input);
}

const questionPaperTool = ai.defineTool({
  name: 'getQuestionPaperText',
  description: 'Extracts the text from a question paper image.',
  inputSchema: z.object({
    questionPaperDataUri: z
      .string()
      .describe(
        "A photo of the question paper, as a data URI that must include a MIME type and use Base64 encoding. Expected format: 'data:<mimetype>;base64,<encoded_data>'."
      ),
  }),
  outputSchema: z.string(),
}, async input => {
  // Dummy implementation
  return "This is the question paper text. Question 1: What is the capital of France? Question 2: Explain the theory of relativity.";
});

const answerKeyTool = ai.defineTool({
  name: 'getAnswerKeyText',
  description: 'Extracts the text from an answer key image.',
  inputSchema: z.object({
    answerKeyDataUri: z
      .string()
      .describe(
        "The answer key, as a data URI that must include a MIME type and use Base64 encoding. Expected format: 'data:<mimetype>;base64,<encoded_data>'."
      ),
  }),
  outputSchema: z.string(),
}, async input => {
  // Dummy implementation
  return "This is the answer key. Answer 1: Paris. Answer 2: Einstein's theory...";
});

const studentAnswerTool = ai.defineTool({
  name: 'getStudentAnswerText',
  description: 'Extracts the text from a student answer sheet image.',
  inputSchema: z.object({
    answerSheetDataUri: z
      .string()
      .describe(
        "The student's answer sheet, as a data URI that must include a MIME type and use Base64 encoding. Expected format: 'data:<mimetype>;base64,<encoded_data>'."
      ),
  }),
  outputSchema: z.string(),
}, async input => {
  // Dummy implementation
  return "This is the student's answer. Answer 1: The capital of France is Paris. Answer 2: I dont know.";
});

const gradeStudentAnswerPrompt = ai.definePrompt({
  name: 'gradeStudentAnswerPrompt',
  input: {
    schema: z.object({
      questionPaperText: z.string().describe('The text extracted from the question paper.'),
      answerKeyText: z.string().describe('The text extracted from the answer key.'),
      studentAnswerText: z.string().describe('The text extracted from the student answer sheet.'),
      marksPerQuestion: z.record(z.number(), z.number()).describe('A dictionary of question numbers to maximum marks for that question.'),
    }),
  },
  output: {
    schema: z.object({
      overallScore: z.number().describe('The overall score of the student.'),
      totalMarks: z.number().describe('The total possible marks.'),
      percentage: z.number().describe('The percentage score of the student.'),
      questionResults: z.record(
        z.object({
          finalScore: z.number().describe('The score awarded for the question.'),
          details: z.record(z.string(), z.number()).optional().describe('Details of the evaluation metrics.'),
          answer_type: z.string().optional().describe('Type of answer')
        })
      ).describe('A dictionary of question numbers to evaluation results.'),
    }),
  },
  prompt: `You are an AI grading assistant. Use the provided information to grade the student's answers.

Question Paper Text: {{{questionPaperText}}}
Answer Key Text: {{{answerKeyText}}}
Student Answer Text: {{{studentAnswerText}}}
Marks per Question: {{{json marksPerQuestion}}}

Provide a detailed breakdown of the marks for each question based on metrics like semantic similarity and coherence.
Return a JSON object with the overall score, total marks, percentage, and question results.
`,
  tools: [questionPaperTool, answerKeyTool, studentAnswerTool]
});

const gradeStudentAnswerFlow = ai.defineFlow<
  typeof GradeStudentAnswerInputSchema,
  typeof GradeStudentAnswerOutputSchema
>({
  name: 'gradeStudentAnswerFlow',
  inputSchema: GradeStudentAnswerInputSchema,
  outputSchema: GradeStudentAnswerOutputSchema,
}, async input => {
  const questionPaperText = await questionPaperTool({
    questionPaperDataUri: input.questionPaperDataUri,
  });
  const answerKeyText = await answerKeyTool({
    answerKeyDataUri: input.answerKeyDataUri,
  });
  const studentAnswerText = await studentAnswerTool({
    answerSheetDataUri: input.answerSheetDataUri,
  });

  const {output} = await gradeStudentAnswerPrompt({
    questionPaperText,
    answerKeyText,
    studentAnswerText,
    marksPerQuestion: input.marksPerQuestion,
  });
  return output!;
});
