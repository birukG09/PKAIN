import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { 
  Plus, 
  Search, 
  Download, 
  Moon, 
  Sun, 
  Lightbulb, 
  Trash2,
  Menu,
  X
} from "lucide-react";
import type { Note } from "@shared/schema";
import { useTheme } from "@/hooks/use-theme";
import { apiRequest } from "@/lib/queryClient";
import { useToast } from "@/hooks/use-toast";
import AISearch from "@/components/ai-search";
import NoteList from "@/components/note-list";

interface SidebarProps {
  notes: Note[];
  isLoading: boolean;
  selectedNoteId: number | null;
  onSelectNote: (id: number) => void;
  onCreateNote: () => void;
  isOpen: boolean;
  onToggle: () => void;
}

export default function Sidebar({
  notes,
  isLoading,
  selectedNoteId,
  onSelectNote,
  onCreateNote,
  isOpen,
  onToggle
}: SidebarProps) {
  const [searchQuery, setSearchQuery] = useState("");
  const { theme, toggleTheme } = useTheme();
  const { toast } = useToast();
  const queryClient = useQueryClient();

  const deleteNoteMutation = useMutation({
    mutationFn: async (id: number) => {
      await apiRequest("DELETE", `/api/notes/${id}`);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/notes"] });
      toast({ title: "Note deleted successfully" });
    },
    onError: () => {
      toast({ 
        title: "Failed to delete note", 
        variant: "destructive" 
      });
    }
  });

  const exportMutation = useMutation({
    mutationFn: async (format: string) => {
      const response = await apiRequest("POST", `/api/export/${format}`, {
        noteIds: notes.map(n => n.id)
      });
      return response.json();
    },
    onSuccess: (data) => {
      toast({ 
        title: `Export completed: ${data.filename}`,
        description: "File saved to downloads"
      });
    },
    onError: () => {
      toast({ 
        title: "Export failed", 
        variant: "destructive" 
      });
    }
  });

  const filteredNotes = notes.filter(note =>
    note.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
    note.content.toLowerCase().includes(searchQuery.toLowerCase()) ||
    note.tags.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase()))
  );

  const handleDelete = (e: React.MouseEvent, id: number) => {
    e.stopPropagation();
    deleteNoteMutation.mutate(id);
  };

  if (!isOpen) {
    return (
      <Button
        variant="ghost"
        size="icon"
        onClick={onToggle}
        className="fixed top-4 left-4 z-50 bg-white dark:bg-slate-800 shadow-lg"
      >
        <Menu className="h-4 w-4" />
      </Button>
    );
  }

  return (
    <div className="w-80 glass-effect border-r border-border/20 flex flex-col transition-all duration-300">
      {/* Header */}
      <div className="p-4 border-b border-border/20 flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <div className="w-8 h-8 bg-gradient-to-br from-primary to-secondary rounded-lg flex items-center justify-center neon-glow">
            <Lightbulb className="w-5 h-5 text-white" />
          </div>
          <h1 className="text-lg font-semibold text-foreground">PKAIN</h1>
        </div>
        
        <div className="flex items-center space-x-1">
          <Button
            variant="ghost"
            size="icon"
            onClick={toggleTheme}
            className="text-muted-foreground hover:text-primary hover:bg-primary/10"
          >
            {theme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
          </Button>
          
          <Button
            variant="ghost"
            size="icon"
            onClick={onToggle}
            className="text-muted-foreground hover:text-primary hover:bg-primary/10"
          >
            <X className="h-4 w-4" />
          </Button>
        </div>
      </div>

      {/* AI Search */}
      <div className="p-4 border-b border-border/20">
        <AISearch />
      </div>

      {/* Quick Actions */}
      <div className="p-4 border-b border-border/20">
        <div className="flex space-x-2">
          <Button
            onClick={onCreateNote}
            className="flex-1 bg-gradient-to-r from-primary to-secondary text-white hover:opacity-90 neon-glow"
          >
            <Plus className="w-4 h-4 mr-2" />
            New Note
          </Button>
          
          <Button
            variant="outline"
            size="icon"
            onClick={() => exportMutation.mutate("pdf")}
            disabled={exportMutation.isPending}
            className="border-primary/30 hover:bg-primary/10"
          >
            <Download className="w-4 h-4" />
          </Button>
        </div>
      </div>

      {/* Search */}
      <div className="p-4 border-b border-border/20">
        <div className="relative">
          <Search className="w-4 h-4 text-muted-foreground absolute left-3 top-2.5" />
          <Input
            placeholder="Search notes..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-10 glass-effect border-primary/20 input-contrast"
          />
        </div>
      </div>

      {/* Notes List */}
      <div className="flex-1 overflow-y-auto">
        <div className="p-4">
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-sm font-medium text-foreground">
              Recent Notes
            </h2>
            <span className="text-xs text-muted-foreground">
              {filteredNotes.length} notes
            </span>
          </div>
          
          <NoteList
            notes={filteredNotes}
            selectedNoteId={selectedNoteId}
            onSelectNote={onSelectNote}
            onDeleteNote={handleDelete}
            isLoading={isLoading}
          />
        </div>
      </div>
    </div>
  );
}
