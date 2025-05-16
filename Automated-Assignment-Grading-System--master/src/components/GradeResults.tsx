
import {Card, CardContent, CardDescription, CardHeader, CardTitle} from '@/components/ui/card';

interface GradeResultsProps {
  results: any;
}

export const GradeResults: React.FC<GradeResultsProps> = ({results}) => {
  return (
    <div className="mt-8 w-full max-w-2xl p-4 rounded-lg shadow-md bg-secondary text-foreground">
      <h2 className="text-2xl font-semibold mb-4 text-primary">Grading Results</h2>

      <Card className="bg-muted">
        <CardHeader>
          <CardTitle className="text-lg font-semibold text-accent-foreground">Overall Performance</CardTitle>
          <CardDescription className="text-sm text-muted-foreground">Summary of the student&apos;s performance.</CardDescription>
        </CardHeader>
        <CardContent className="text-accent-foreground">
          <p className="text-lg">Overall Score: {results.overallScore} / {results.totalMarks}</p>
          <p className="text-lg">Percentage: {results.percentage}%</p>
        </CardContent>
      </Card>

      <div className="mt-6">
        <h3 className="text-xl font-semibold mb-2 text-primary">Marks Breakage</h3>
        {Object.entries(results.questionResults).map(([question, result]: [string, any]) => (
          <Card key={question} className="mb-4 bg-muted">
            <CardHeader>
              <CardTitle className="text-base font-semibold text-accent-foreground">Question {question}</CardTitle>
              <CardDescription className="text-sm text-muted-foreground">Detailed metrics for question {question}.</CardDescription>
            </CardHeader>
            <CardContent className="text-accent-foreground">
              <p>Final Score: {result.finalScore}</p>
              {result.details && (
                <>
                  <h4 className="text-md font-semibold mt-2 text-accent-foreground">Details</h4>
                  <ul>
                    {Object.entries(result.details).map(([metric, score]: [string, number]) => (
                      <li key={metric}>
                        {metric}: {score.toFixed(3)}
                      </li>
                    ))}
                  </ul>
                </>
              )}
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
};

    