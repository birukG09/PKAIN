import { notes, embeddings, type Note, type InsertNote, type UpdateNote, type Embedding } from "@shared/schema";

export interface IStorage {
  // Note operations
  getNotes(): Promise<Note[]>;
  getNote(id: number): Promise<Note | undefined>;
  createNote(note: InsertNote): Promise<Note>;
  updateNote(id: number, note: UpdateNote): Promise<Note | undefined>;
  deleteNote(id: number): Promise<boolean>;
  searchNotesByText(query: string): Promise<Note[]>;
  
  // Embedding operations
  getEmbeddings(noteId: number): Promise<Embedding[]>;
  createEmbedding(noteId: number, embedding: number[], chunkText: string, chunkIndex: number): Promise<Embedding>;
  deleteEmbeddings(noteId: number): Promise<boolean>;
  getAllEmbeddings(): Promise<Embedding[]>;
}

export class MemStorage implements IStorage {
  private notes: Map<number, Note>;
  private embeddings: Map<number, Embedding>;
  private currentNoteId: number;
  private currentEmbeddingId: number;

  constructor() {
    this.notes = new Map();
    this.embeddings = new Map();
    this.currentNoteId = 1;
    this.currentEmbeddingId = 1;
  }

  async getNotes(): Promise<Note[]> {
    return Array.from(this.notes.values()).sort((a, b) => 
      new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime()
    );
  }

  async getNote(id: number): Promise<Note | undefined> {
    return this.notes.get(id);
  }

  async createNote(insertNote: InsertNote): Promise<Note> {
    const id = this.currentNoteId++;
    const now = new Date().toISOString();
    const note: Note = {
      ...insertNote,
      id,
      createdAt: now,
      updatedAt: now,
    };
    this.notes.set(id, note);
    return note;
  }

  async updateNote(id: number, updateNote: UpdateNote): Promise<Note | undefined> {
    const existingNote = this.notes.get(id);
    if (!existingNote) return undefined;

    const updatedNote: Note = {
      ...existingNote,
      ...updateNote,
      updatedAt: new Date().toISOString(),
    };
    this.notes.set(id, updatedNote);
    return updatedNote;
  }

  async deleteNote(id: number): Promise<boolean> {
    const deleted = this.notes.delete(id);
    if (deleted) {
      // Also delete associated embeddings
      await this.deleteEmbeddings(id);
    }
    return deleted;
  }

  async searchNotesByText(query: string): Promise<Note[]> {
    const searchTerm = query.toLowerCase();
    return Array.from(this.notes.values()).filter(note =>
      note.title.toLowerCase().includes(searchTerm) ||
      note.content.toLowerCase().includes(searchTerm) ||
      note.tags.some(tag => tag.toLowerCase().includes(searchTerm))
    );
  }

  async getEmbeddings(noteId: number): Promise<Embedding[]> {
    return Array.from(this.embeddings.values()).filter(emb => emb.noteId === noteId);
  }

  async createEmbedding(noteId: number, embedding: number[], chunkText: string, chunkIndex: number): Promise<Embedding> {
    const id = this.currentEmbeddingId++;
    const embeddingRecord: Embedding = {
      id,
      noteId,
      embedding: JSON.stringify(embedding),
      chunkText,
      chunkIndex,
    };
    this.embeddings.set(id, embeddingRecord);
    return embeddingRecord;
  }

  async deleteEmbeddings(noteId: number): Promise<boolean> {
    const embeddingsToDelete = Array.from(this.embeddings.entries())
      .filter(([, emb]) => emb.noteId === noteId);
    
    embeddingsToDelete.forEach(([id]) => this.embeddings.delete(id));
    return embeddingsToDelete.length > 0;
  }

  async getAllEmbeddings(): Promise<Embedding[]> {
    return Array.from(this.embeddings.values());
  }
}

export const storage = new MemStorage();
