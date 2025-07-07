import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { Trash2 } from "lucide-react";
import type { Note } from "@shared/schema";
import { formatDistanceToNow } from "date-fns";

interface NoteListProps {
  notes: Note[];
  selectedNoteId: number | null;
  onSelectNote: (id: number) => void;
  onDeleteNote: (e: React.MouseEvent, id: number) => void;
  isLoading: boolean;
}

export default function NoteList({
  notes,
  selectedNoteId,
  onSelectNote,
  onDeleteNote,
  isLoading
}: NoteListProps) {
  if (isLoading) {
    return (
      <div className="space-y-2">
        {[...Array(3)].map((_, i) => (
          <Card key={i} className="p-3">
            <CardContent className="p-0">
              <Skeleton className="h-4 w-3/4 mb-2" />
              <Skeleton className="h-3 w-full mb-2" />
              <div className="flex space-x-2">
                <Skeleton className="h-5 w-12" />
                <Skeleton className="h-3 w-16" />
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    );
  }

  if (notes.length === 0) {
    return (
      <div className="text-center py-8">
        <p className="text-muted-foreground text-sm">
          No notes found. Create your first note to get started!
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {notes.map((note) => (
        <Card
          key={note.id}
          className={`cursor-pointer transition-all duration-300 group hover:neon-glow ${
            selectedNoteId === note.id
              ? 'glass-effect border-primary/50 neon-glow'
              : 'glass-effect hover:border-primary/30'
          }`}
          onClick={() => onSelectNote(note.id)}
        >
          <CardContent className="p-3">
            <div className="flex items-start justify-between">
              <div className="flex-1 min-w-0">
                <h3 className="text-sm font-medium text-foreground truncate">
                  {note.title || "Untitled"}
                </h3>
                <p className="text-xs text-muted-foreground mt-1 line-clamp-2">
                  {note.content.replace(/[#*`]/g, '').substring(0, 100)}
                  {note.content.length > 100 && '...'}
                </p>
                <div className="flex items-center mt-2 space-x-2">
                  {note.tags.slice(0, 2).map((tag, index) => (
                    <Badge 
                      key={index} 
                      variant="secondary" 
                      className="text-xs"
                    >
                      {tag}
                    </Badge>
                  ))}
                  {note.tags.length > 2 && (
                    <Badge variant="outline" className="text-xs">
                      +{note.tags.length - 2}
                    </Badge>
                  )}
                  <span className="text-xs text-muted-foreground">
                    {formatDistanceToNow(new Date(note.updatedAt), { addSuffix: true })}
                  </span>
                </div>
              </div>
              <div className="ml-2 opacity-0 group-hover:opacity-100 transition-opacity">
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={(e) => onDeleteNote(e, note.id)}
                  className="h-6 w-6 text-muted-foreground hover:text-destructive"
                >
                  <Trash2 className="w-3 h-3" />
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}
