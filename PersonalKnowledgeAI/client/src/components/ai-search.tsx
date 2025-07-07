import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Search, Brain, Loader2 } from "lucide-react";
import { apiRequest } from "@/lib/queryClient";
import type { AIResponse } from "@shared/schema";

export default function AISearch() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<AIResponse | null>(null);
  const [showResults, setShowResults] = useState(false);

  const aiSearchMutation = useMutation({
    mutationFn: async (searchQuery: string) => {
      const response = await apiRequest("POST", "/api/ai/query", { 
        query: searchQuery 
      });
      return response.json() as Promise<AIResponse>;
    },
    onSuccess: (data) => {
      setResults(data);
      setShowResults(true);
    },
  });

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim()) {
      aiSearchMutation.mutate(query);
    }
  };

  return (
    <div className="space-y-4">
      <form onSubmit={handleSearch} className="relative">
        <Search className="w-4 h-4 text-slate-400 absolute left-3 top-2.5" />
        <Input
          placeholder="Ask AI about your notes..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          className="pl-10 pr-12 input-contrast"
        />
        <Button
          type="submit"
          size="icon"
          variant="ghost"
          className="absolute right-1 top-1 h-6 w-6"
          disabled={aiSearchMutation.isPending || !query.trim()}
        >
          {aiSearchMutation.isPending ? (
            <Loader2 className="h-3 w-3 animate-spin" />
          ) : (
            <Brain className="h-3 w-3" />
          )}
        </Button>
      </form>

      {showResults && results && (
        <Card className="glass-effect neon-glow">
          <CardContent className="p-4">
            <div className="flex items-center gap-2 mb-3">
              <Brain className="h-4 w-4 text-primary" />
              <span className="text-sm font-medium">AI Answer</span>
              <Badge variant="secondary" className="text-xs bg-primary/20 text-primary">
                {Math.round(results.confidence * 100)}% confidence
              </Badge>
            </div>
            
            <p className="text-sm text-foreground mb-3">
              {results.answer}
            </p>

            {results.sources.length > 0 && (
              <div>
                <p className="text-xs text-muted-foreground mb-2">Sources:</p>
                <div className="space-y-2">
                  {results.sources.slice(0, 3).map((source, index) => (
                    <div key={index} className="text-xs p-2 glass-effect rounded border border-primary/20">
                      <div className="font-medium text-foreground">
                        {source.note.title}
                      </div>
                      <div className="text-muted-foreground mt-1">
                        {source.matchedText.substring(0, 100)}...
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            <Button
              variant="ghost"
              size="sm"
              onClick={() => setShowResults(false)}
              className="mt-3 w-full text-xs"
            >
              Close
            </Button>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
