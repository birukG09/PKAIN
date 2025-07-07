import { apiRequest } from "./queryClient";
import type { Note, InsertNote, UpdateNote, AIQuery, AIResponse, SearchResult } from "@shared/schema";

export const api = {
  // Notes
  getNotes: async (): Promise<Note[]> => {
    const response = await fetch("/api/notes");
    return response.json();
  },

  getNote: async (id: number): Promise<Note> => {
    const response = await fetch(`/api/notes/${id}`);
    return response.json();
  },

  createNote: async (note: InsertNote): Promise<Note> => {
    const response = await apiRequest("POST", "/api/notes", note);
    return response.json();
  },

  updateNote: async (id: number, note: UpdateNote): Promise<Note> => {
    const response = await apiRequest("PATCH", `/api/notes/${id}`, note);
    return response.json();
  },

  deleteNote: async (id: number): Promise<void> => {
    await apiRequest("DELETE", `/api/notes/${id}`);
  },

  // Search
  searchNotes: async (query: string): Promise<Note[]> => {
    const response = await fetch(`/api/search?q=${encodeURIComponent(query)}`);
    return response.json();
  },

  semanticSearch: async (query: string): Promise<SearchResult[]> => {
    const response = await apiRequest("POST", "/api/search/semantic", { query });
    return response.json();
  },

  // AI
  aiQuery: async (query: AIQuery): Promise<AIResponse> => {
    const response = await apiRequest("POST", "/api/ai/query", query);
    return response.json();
  },

  // Export
  exportNotes: async (noteIds: number[], format: string): Promise<{ filename: string; path: string }> => {
    const response = await apiRequest("POST", `/api/export/${format}`, { noteIds });
    return response.json();
  }
};
