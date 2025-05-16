// use server'
'use server';
/**
 * @fileOverview An answer key suggestion AI agent.
 *
 * - suggestAnswerKey - A function that handles the answer key suggestion process.
 * - SuggestAnswerKeyInput - The input type for the suggestAnswerKey function.
 * - SuggestAnswerKeyOutput - The return type for the suggestAnswerKey function.
 */

import {ai} from '@/ai/ai-instance';
import {z} from 'genkit';

const SuggestAnswerKeyInputSchema = z.object({
  questionPaperDataUri: z
    .string()
    .describe(
      "A question paper, as a data URI that must include a MIME type and use Base64 encoding. Expected format: 'data:<mimetype>;base64,<encoded_data>'."
    ),
});
export type SuggestAnswerKeyInput = z.infer<typeof SuggestAnswerKeyInputSchema>;

const SuggestAnswerKeyOutputSchema = z.object({
  suggestedAnswerKey: z.record(z.string(), z.string()).describe('A dictionary of suggested answers for each question.'),
});
export type SuggestAnswerKeyOutput = z.infer<typeof SuggestAnswerKeyOutputSchema>;

export async function suggestAnswerKey(input: SuggestAnswerKeyInput): Promise<SuggestAnswerKeyOutput> {
  return suggestAnswerKeyFlow(input);
}

const prompt = ai.definePrompt({
  name: 'suggestAnswerKeyPrompt',
  input: {
    schema: z.object({
      questionPaperDataUri: z
        .string()
        .describe(
          "A question paper, as a data URI that must include a MIME type and use Base64 encoding. Expected format: 'data:<mimetype>;base64,<encoded_data>'."
        ),
    }),
  },
  output: {
    schema: z.object({
      suggestedAnswerKey: z.record(z.string(), z.string()).describe('A dictionary of suggested answers for each question.'),
    }),
  },
  prompt: `You are an expert teacher specializing in creating answer keys for question papers.\n\nYou will use the provided question paper to create a suggested answer key.\nThe answer key should be a dictionary where the keys are question numbers and the values are the suggested answers.\n\nQuestion Paper: {{media url=questionPaperDataUri}}\n\nFormat the answer key as a JSON object.\n`,
});

const suggestAnswerKeyFlow = ai.defineFlow<
  typeof SuggestAnswerKeyInputSchema,
  typeof SuggestAnswerKeyOutputSchema
>({
  name: 'suggestAnswerKeyFlow',
  inputSchema: SuggestAnswerKeyInputSchema,
  outputSchema: SuggestAnswerKeyOutputSchema,
}, async input => {
  const {output} = await prompt(input);
  return output!;
});
