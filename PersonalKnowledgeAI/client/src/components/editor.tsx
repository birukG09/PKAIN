import { useState, useEffect } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent } from "@/components/ui/card";
import { 
  Save, 
  Eye, 
  Edit3, 
  Columns, 
  Plus, 
  X, 
  CheckCircle,
  Loader2,
  Brain,
  Sparkles
} from "lucide-react";
import ReactMarkdown from "react-markdown";
import type { Note, InsertNote, UpdateNote } from "@shared/schema";
import { apiRequest } from "@/lib/queryClient";
import { useToast } from "@/hooks/use-toast";

interface EditorProps {
  note: Note | null;
  onSave: (note: Note) => void;
}

export default function Editor({ note, onSave }: EditorProps) {
  const [title, setTitle] = useState("");
  const [content, setContent] = useState("");
  const [tags, setTags] = useState<string[]>([]);
  const [newTag, setNewTag] = useState("");
  const [viewMode, setViewMode] = useState<"edit" | "preview" | "split">("edit");
  const [hasChanges, setHasChanges] = useState(false);
  const [summary, setSummary] = useState<any>(null);
  const [showSummary, setShowSummary] = useState(false);
  
  const { toast } = useToast();
  const queryClient = useQueryClient();

  // Load note data when note changes
  useEffect(() => {
    if (note) {
      setTitle(note.title);
      setContent(note.content);
      setTags(note.tags);
      setHasChanges(false);
    } else {
      setTitle("");
      setContent("");
      setTags([]);
      setHasChanges(false);
    }
  }, [note]);

  // Track changes
  useEffect(() => {
    if (note) {
      const changed = 
        title !== note.title ||
        content !== note.content ||
        JSON.stringify(tags) !== JSON.stringify(note.tags);
      setHasChanges(changed);
    } else {
      setHasChanges(title.trim() !== "" || content.trim() !== "" || tags.length > 0);
    }
  }, [title, content, tags, note]);

  const createNoteMutation = useMutation({
    mutationFn: async (noteData: InsertNote) => {
      const response = await apiRequest("POST", "/api/notes", noteData);
      return response.json() as Promise<Note>;
    },
    onSuccess: (newNote) => {
      queryClient.invalidateQueries({ queryKey: ["/api/notes"] });
      onSave(newNote);
      setHasChanges(false);
      toast({ title: "Note created successfully" });
    },
    onError: () => {
      toast({ 
        title: "Failed to create note", 
        variant: "destructive" 
      });
    }
  });

  const updateNoteMutation = useMutation({
    mutationFn: async (data: { id: number; note: UpdateNote }) => {
      const response = await apiRequest("PATCH", `/api/notes/${data.id}`, data.note);
      return response.json() as Promise<Note>;
    },
    onSuccess: (updatedNote) => {
      queryClient.invalidateQueries({ queryKey: ["/api/notes"] });
      onSave(updatedNote);
      setHasChanges(false);
      toast({ title: "Note saved successfully" });
    },
    onError: () => {
      toast({ 
        title: "Failed to save note", 
        variant: "destructive" 
      });
    }
  });

  // AI Summarization mutation
  const summarizeMutation = useMutation({
    mutationFn: async ({ noteId, content }: { noteId?: number, content: string }) => {
      const response = await apiRequest("POST", "/api/ai/summarize", {
        noteId,
        content,
        summaryType: "auto"
      });
      return response.json();
    },
    onSuccess: (data) => {
      setSummary(data);
      setShowSummary(true);
      toast({
        title: "TL;DR Generated",
        description: "AI summary has been created successfully."
      });
    },
    onError: (error) => {
      console.error("Summarization error:", error);
      toast({
        title: "Summarization Failed",
        description: "Unable to generate summary. Please try again.",
        variant: "destructive"
      });
    }
  });

  const handleSave = () => {
    const noteData = {
      title: title.trim() || "Untitled",
      content: content.trim(),
      tags: tags.filter(tag => tag.trim() !== "")
    };

    if (note) {
      updateNoteMutation.mutate({ id: note.id, note: noteData });
    } else {
      createNoteMutation.mutate(noteData);
    }
  };

  const handleSummarize = () => {
    if (!content.trim()) {
      toast({
        title: "No Content",
        description: "Please add some content to summarize.",
        variant: "destructive"
      });
      return;
    }

    summarizeMutation.mutate({
      noteId: note?.id,
      content: content.trim()
    });
  };

  const handleAddTag = () => {
    if (newTag.trim() && !tags.includes(newTag.trim())) {
      setTags([...tags, newTag.trim()]);
      setNewTag("");
    }
  };

  const handleRemoveTag = (tagToRemove: string) => {
    setTags(tags.filter(tag => tag !== tagToRemove));
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && e.ctrlKey) {
      handleSave();
    }
  };

  const isSaving = createNoteMutation.isPending || updateNoteMutation.isPending;

  const renderMarkdown = () => (
    <div className="prose prose-slate dark:prose-invert max-w-none">
      <ReactMarkdown>{content || "Start writing your note..."}</ReactMarkdown>
    </div>
  );

  if (!note && !hasChanges) {
    return (
      <div className="flex-1 flex items-center justify-center glass-effect">
        <Card className="w-full max-w-md mx-4 glass-effect neon-glow">
          <CardContent className="pt-6 text-center">
            <div className="w-16 h-16 mx-auto mb-4 rounded-2xl bg-gradient-to-br from-primary to-secondary flex items-center justify-center floating-animation">
              <Edit3 className="h-8 w-8 text-white" />
            </div>
            <h2 className="text-xl font-semibold text-foreground mb-2">
              Welcome to PKAIN
            </h2>
            <p className="text-muted-foreground mb-4">
              Your AI-powered personal knowledge notebook. Create a new note or select an existing one to get started.
            </p>
            <Button 
              onClick={() => setTitle("New Note")} 
              className="bg-gradient-to-r from-primary to-secondary text-white hover:opacity-90 neon-glow"
            >
              <Plus className="w-4 h-4 mr-2" />
              Create Your First Note
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="flex-1 flex flex-col glass-effect">
      {/* Header */}
      <div className="p-4 border-b border-border/20 flex items-center justify-between glass-effect">
        <div className="flex items-center space-x-4 flex-1 min-w-0">
          <Input
            placeholder="Untitled Note"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            className="text-lg font-semibold bg-transparent border-none outline-none focus-visible:ring-0 px-0"
            style={{ color: 'hsl(220, 20%, 15%)', caretColor: 'hsl(220, 20%, 15%)' }}
            onKeyPress={handleKeyPress}
          />
          
          {/* Tags */}
          <div className="flex items-center space-x-2 flex-wrap">
            {tags.map((tag, index) => (
              <Badge 
                key={index} 
                variant="secondary" 
                className="flex items-center gap-1"
              >
                {tag}
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => handleRemoveTag(tag)}
                  className="h-3 w-3 p-0 hover:bg-transparent"
                >
                  <X className="w-2 h-2" />
                </Button>
              </Badge>
            ))}
            
            <div className="flex items-center space-x-2">
              <Input
                placeholder="Add tag..."
                value={newTag}
                onChange={(e) => setNewTag(e.target.value)}
                onKeyPress={(e) => e.key === "Enter" && handleAddTag()}
                className="w-20 h-6 text-xs input-contrast"
              />
              <Button
                size="icon"
                variant="ghost"
                onClick={handleAddTag}
                className="h-6 w-6"
              >
                <Plus className="w-3 h-3" />
              </Button>
            </div>
          </div>
        </div>
        
        <div className="flex items-center space-x-2">
          {/* View Toggle */}
          <Tabs value={viewMode} onValueChange={(value) => setViewMode(value as any)}>
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="edit" className="text-xs">
                <Edit3 className="w-3 h-3 mr-1" />
                Edit
              </TabsTrigger>
              <TabsTrigger value="preview" className="text-xs">
                <Eye className="w-3 h-3 mr-1" />
                Preview
              </TabsTrigger>
              <TabsTrigger value="split" className="text-xs">
                <Columns className="w-3 h-3 mr-1" />
                Split
              </TabsTrigger>
            </TabsList>
          </Tabs>
          
          {/* Save Button */}
          <Button
            onClick={handleSave}
            disabled={!hasChanges || isSaving}
            className="bg-gradient-to-r from-primary to-secondary text-white hover:opacity-90 neon-glow"
          >
            {isSaving ? (
              <Loader2 className="w-4 h-4 mr-2 animate-spin" />
            ) : hasChanges ? (
              <Save className="w-4 h-4 mr-2" />
            ) : (
              <CheckCircle className="w-4 h-4 mr-2 text-green-500" />
            )}
            {isSaving ? "Saving..." : hasChanges ? "Save" : "Saved"}
          </Button>
          
          {/* TL;DR Button */}
          <Button
            onClick={handleSummarize}
            disabled={!content.trim() || summarizeMutation.isPending}
            variant="outline"
            className="border-primary/20 hover:bg-primary/10 neon-glow"
          >
            {summarizeMutation.isPending ? (
              <Loader2 className="w-4 h-4 mr-2 animate-spin" />
            ) : (
              <Brain className="w-4 h-4 mr-2" />
            )}
            {summarizeMutation.isPending ? "Generating..." : "TL;DR"}
          </Button>
        </div>
      </div>

      {/* Editor Content */}
      <div className="flex-1 flex">
        {viewMode === "edit" && (
          <div className="flex-1 p-4">
            <Textarea
              placeholder="Start writing your note in Markdown..."
              value={content}
              onChange={(e) => setContent(e.target.value)}
              onKeyDown={handleKeyPress}
              className="w-full h-full resize-none border-none outline-none font-mono text-sm leading-relaxed focus-visible:ring-0"
              style={{ color: 'hsl(220, 20%, 15%)', caretColor: 'hsl(220, 20%, 15%)' }}
            />
          </div>
        )}
        
        {viewMode === "preview" && (
          <div className="flex-1 p-4 overflow-y-auto">
            {renderMarkdown()}
          </div>
        )}
        
        {viewMode === "split" && (
          <>
            <div className="flex-1 p-4 border-r border-border/20">
              <Textarea
                placeholder="Start writing your note in Markdown..."
                value={content}
                onChange={(e) => setContent(e.target.value)}
                onKeyDown={handleKeyPress}
                className="w-full h-full resize-none border-none outline-none font-mono text-sm leading-relaxed focus-visible:ring-0"
                style={{ color: 'hsl(220, 20%, 15%)', caretColor: 'hsl(220, 20%, 15%)' }}
              />
            </div>
            <div className="flex-1 p-4 overflow-y-auto">
              {renderMarkdown()}
            </div>
          </>
        )}
      </div>
      
      {/* AI Summary Display */}
      {showSummary && summary && (
        <div className="p-4 border-t border-border/20 glass-effect">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center space-x-2">
              <Sparkles className="w-4 h-4 text-primary" />
              <h3 className="font-semibold text-foreground">TL;DR Summary</h3>
            </div>
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setShowSummary(false)}
              className="h-6 w-6"
            >
              <X className="w-4 h-4" />
            </Button>
          </div>
          
          <div className="space-y-4">
            <div className="bg-background/50 p-3 rounded-lg border border-border/10">
              <h4 className="text-sm font-medium text-foreground mb-2">Summary</h4>
              <p className="text-sm text-muted-foreground leading-relaxed">
                {summary.summary}
              </p>
            </div>
            
            {summary.keyPoints && summary.keyPoints.length > 0 && (
              <div className="bg-background/50 p-3 rounded-lg border border-border/10">
                <h4 className="text-sm font-medium text-foreground mb-2">Key Points</h4>
                <ul className="text-sm text-muted-foreground space-y-1">
                  {summary.keyPoints.map((point: string, index: number) => (
                    <li key={index} className="flex items-start space-x-2">
                      <span className="text-primary">•</span>
                      <span>{point}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
            
            <div className="flex items-center space-x-4 text-xs text-muted-foreground">
              <span>Words: {summary.wordCount}</span>
              <span>Reading Time: ~{summary.readingTime} min</span>
              <span>Confidence: {Math.round(summary.confidence * 100)}%</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
