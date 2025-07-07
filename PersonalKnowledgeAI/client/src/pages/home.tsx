import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import type { Note } from "@shared/schema";
import Sidebar from "@/components/sidebar";
import Editor from "@/components/editor";

export default function Home() {
  const [selectedNoteId, setSelectedNoteId] = useState<number | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(true);

  const { data: notes = [], isLoading } = useQuery<Note[]>({
    queryKey: ['/api/notes'],
  });

  const selectedNote = selectedNoteId 
    ? notes.find(note => note.id === selectedNoteId) 
    : null;

  return (
    <div className="flex h-screen overflow-hidden matrix-grid">
      <Sidebar
        notes={notes}
        isLoading={isLoading}
        selectedNoteId={selectedNoteId}
        onSelectNote={setSelectedNoteId}
        onCreateNote={() => setSelectedNoteId(null)}
        isOpen={sidebarOpen}
        onToggle={() => setSidebarOpen(!sidebarOpen)}
      />
      
      <div className="flex-1 flex flex-col">
        <Editor
          note={selectedNote}
          onSave={(note) => {
            if (note.id && !selectedNoteId) {
              setSelectedNoteId(note.id);
            }
          }}
        />
      </div>
    </div>
  );
}
