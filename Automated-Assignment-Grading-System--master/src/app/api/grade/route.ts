import {gradeStudentAnswer} from '@/ai/flows/grade-student-answer';
import {NextResponse} from 'next/server';

export async function POST(req: Request) {
  try {
    const {questionPaperDataUri, answerSheetDataUri, answerKeyDataUri} = await req.json();

    // Call the AI grading function
    const results = await gradeStudentAnswer({
      questionPaperDataUri,
      answerSheetDataUri,
      answerKeyDataUri,
      marksPerQuestion: {
        1: 3,
        2: 7,
        3: 5,
      },
    });

    // Return the grading results
    return NextResponse.json(results);
  } catch (error) {
    console.error('API grading error:', error);
    return NextResponse.json({error: 'Failed to grade student answer'}, {status: 500});
  }
}
