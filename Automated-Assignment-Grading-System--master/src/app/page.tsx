
'use client';

import {GradeForm} from '@/components/GradeForm';
import {GradeResults} from '@/components/GradeResults';
import {useState} from 'react';

export default function Home() {
  const [gradeResults, setGradeResults] = useState(null);

  return (
    <main className="flex min-h-screen flex-col items-center justify-start p-8 bg-background text-foreground">
      <h1 className="text-4xl font-bold mb-8">Automated Assignment Grading System</h1>
      <GradeForm setGradeResults={setGradeResults} />
      {gradeResults && <GradeResults results={gradeResults} />}
    </main>
  );
}

    