import { pgTable, text, serial, integer, boolean, timestamp } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";

export const notes = pgTable("notes", {
  id: serial("id").primaryKey(),
  title: text("title").notNull(),
  content: text("content").notNull(),
  tags: text("tags").array().notNull().default([]),
  createdAt: timestamp("created_at").defaultNow().notNull(),
  updatedAt: timestamp("updated_at").defaultNow().notNull(),
});

export const insertNoteSchema = createInsertSchema(notes).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export const updateNoteSchema = createInsertSchema(notes).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
}).partial();

export type InsertNote = z.infer<typeof insertNoteSchema>;
export type UpdateNote = z.infer<typeof updateNoteSchema>;
export type Note = typeof notes.$inferSelect;

// AI Search and embeddings
export const embeddings = pgTable("embeddings", {
  id: serial("id").primaryKey(),
  noteId: integer("note_id").references(() => notes.id).notNull(),
  embedding: text("embedding").notNull(), // JSON serialized vector
  chunkText: text("chunk_text").notNull(),
  chunkIndex: integer("chunk_index").notNull(),
});

export type Embedding = typeof embeddings.$inferSelect;

// Search results type
export const searchResultSchema = z.object({
  note: z.object({
    id: z.number(),
    title: z.string(),
    content: z.string(),
    tags: z.array(z.string()),
    createdAt: z.string(),
    updatedAt: z.string(),
  }),
  score: z.number(),
  matchedText: z.string(),
});

export type SearchResult = z.infer<typeof searchResultSchema>;

// AI Query types
export const aiQuerySchema = z.object({
  query: z.string().min(1),
  maxResults: z.number().optional().default(5),
});

export type AIQuery = z.infer<typeof aiQuerySchema>;

export const aiResponseSchema = z.object({
  answer: z.string(),
  sources: z.array(searchResultSchema),
  confidence: z.number(),
});

export type AIResponse = z.infer<typeof aiResponseSchema>;
