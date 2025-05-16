'use client';

import {Button} from '@/components/ui/button';
import {Input} from '@/components/ui/input';
import {useState} from 'react';
import {toast} from '@/hooks/use-toast';
import {AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle, AlertDialogTrigger} from '@/components/ui/alert-dialog';
import {Textarea} from '@/components/ui/textarea';
import {File, Upload, BookOpen, MessageSquare, Loader2} from 'lucide-react';

interface GradeFormProps {
  setGradeResults: (results: any) => void;
}

export const GradeForm: React.FC<GradeFormProps> = ({setGradeResults}) => {
  const [questionPaper, setQuestionPaper] = useState<File | null>(null);
  const [answerSheet, setAnswerSheet] = useState<File | null>(null);
  const [answerSheetName, setAnswerSheetName] = useState<string | null>(null);
  const [answerKey, setAnswerKey] = useState<File | null>(null);
  const [answerKeyText, setAnswerKeyText] = useState(''); // State for text input
  const [rubricsText, setRubricsText] = useState(''); // State for text input
  const [isLoading, setIsLoading] = useState(false);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>, setter: (file: File | null) => void, nameSetter: (name: string | null) => void) => {
    const file = event.target.files?.[0] || null;
    setter(file);
    nameSetter(file ? file.name : null);
  };

  const handleTextChange = (event: React.ChangeEvent<HTMLTextAreaElement>, setter: (text: string) => void) => {
    setter(event.target.value);
  };

  const handleSubmit = async () => {
    if (!questionPaper || !answerSheet || !answerKey) {
      toast({
        title: 'Error',
        description: 'Please upload all the required files.',
        variant: 'destructive',
      });
      return;
    }
  
    setIsLoading(true);
  
    // Create FormData to send files
    const formData = new FormData();
    formData.append('question_paper', questionPaper);
    formData.append('answer_sheet', answerSheet);
    formData.append('answer_key', answerKey);
  
    try {
      const response = await fetch('http://127.0.0.1:5000/grade', {
        method: 'POST',
        body: formData, // Send FormData directly
      });
  
      if (!response.ok) {
        console.error('Server error:', response.status, response.statusText);
        let errorMessage = `Grading failed. Server responded with ${response.status} ${response.statusText}.`;
        try {
          const errorBody = await response.json();
          if (errorBody && errorBody.error) {
            errorMessage += ` Details: ${errorBody.error}`;
          }
        } catch (parseError) {
          console.error('Failed to parse error body:', parseError);
          errorMessage += ' Failed to parse detailed error message.';
        }
        toast({
          title: 'Error',
          description: errorMessage,
          variant: 'destructive',
        });
        setIsLoading(false);
        return;
      }
  
      const data = await response.json();
      console.log(data);
      toast({
        title: 'Success',
        description: 'Grading completed successfully!',
      });
      setGradeResults(data); // Update the state with the grading results
    } catch (error: any) {
      console.error('Client-side error:', error);
      toast({
        title: 'Error',
        description: error.message || 'Failed to grade. Please try again.',
        variant: 'destructive',
      });
    } finally {
      setIsLoading(false);
    }
  };

  // Helper function to convert File to Data URI
  const fileToDataUri = (file: File): Promise<string | null> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = (event) => {
        resolve(event.target?.result as string);
      };
      reader.onerror = (error) => {
        console.error('Error reading file:', error);
        reject(null);
      };
      reader.readAsDataURL(file);
    });
  };

  return (
    <div className="flex flex-col space-y-6 w-full max-w-5xl p-4 rounded-lg shadow-md bg-secondary border border-gray-300">
      <h2 className="text-2xl font-semibold text-primary">Upload Documents</h2>

      {/* Student Answer Sheet Upload */}
      <div className="border-2 border-dashed rounded-md p-8 text-center text-gray-500 flex flex-col items-center justify-center h-[300px] bg-muted transition-colors">
        <Upload className="w-12 h-12 mb-4 mx-auto animate-pulse text-accent-foreground" />
        {answerSheetName ? (
          <p className="text-lg text-accent-foreground">Selected file: {answerSheetName}</p>
        ) : (
          <>
            <p className="text-lg text-accent-foreground">Drag and drop the student's answer PDF here or click to browse</p>
          </>
        )}
        <Input type="file" accept="application/pdf, image/*" onChange={(e) => handleFileChange(e, setAnswerSheet, setAnswerSheetName)} className="hidden" id="answer-sheet-upload" />
        <label htmlFor="answer-sheet-upload" className="mt-4 inline-flex items-center justify-center rounded-md text-sm font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 bg-primary text-primary-foreground  px-4 py-2">
          Select File
        </label>
      </div>

      {/* Question Paper Upload and Answer Key / Rubrics Input */}
      <div className="flex space-x-4">
        <div className="w-1/2 border-2 border-dashed rounded-md p-4 bg-muted flex flex-col items-center">
          <BookOpen className="w-6 h-6 mb-2 text-accent-foreground" />
          <Input type="file" accept="image/*" onChange={(e) => handleFileChange(e, setQuestionPaper, () => {})} placeholder="Upload Question Paper" className="bg-muted text-accent-foreground" />
          <Textarea placeholder="Rubrics" value={rubricsText} onChange={(e) => handleTextChange(e, setRubricsText)} className="mt-2 bg-muted text-accent-foreground" />
        </div>
        <div className="w-1/2 border-2 border-dashed rounded-md p-4 bg-muted flex flex-col items-center">
          <MessageSquare className="w-6 h-6 mb-2 text-accent-foreground" />
          <Input type="file" accept="application/pdf, image/*" onChange={(e) => handleFileChange(e, setAnswerKey, () => {})} placeholder="Upload Answer Key" className="bg-muted text-accent-foreground" />
          <Textarea placeholder="Correct Answers" value={answerKeyText} onChange={(e) => handleTextChange(e, setAnswerKeyText)} className="mt-2 bg-muted text-accent-foreground" />
        </div>
      </div>

      <AlertDialog>
        <AlertDialogTrigger asChild>
          <Button className="bg-primary text-primary-foreground hover:bg-primary/90" disabled={isLoading}>
            {isLoading ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Grading - This may take a while :/
              </>
            ) : (
              'Grade'
            )}
          </Button>
        </AlertDialogTrigger>
        <AlertDialogContent className="bg-secondary text-accent-foreground">
          <AlertDialogHeader>
            <AlertDialogTitle>Are you sure?</AlertDialogTitle>
            <AlertDialogDescription>
              This will initiate the grading process.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel className="bg-muted hover:bg-accent text-accent-foreground">Cancel</AlertDialogCancel>
            <AlertDialogAction onClick={handleSubmit} className="bg-primary text-primary-foreground hover:bg-primary/90" disabled={isLoading}>Continue</AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
};
