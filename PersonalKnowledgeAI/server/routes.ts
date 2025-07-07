import type { Express } from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";
import { insertNoteSchema, updateNoteSchema, aiQuerySchema } from "@shared/schema";
import { spawn } from "child_process";
import path from "path";

export async function registerRoutes(app: Express): Promise<Server> {
  // Notes CRUD
  app.get("/api/notes", async (req, res) => {
    try {
      const notes = await storage.getNotes();
      res.json(notes);
    } catch (error) {
      res.status(500).json({ message: "Failed to fetch notes" });
    }
  });

  app.get("/api/notes/:id", async (req, res) => {
    try {
      const id = parseInt(req.params.id);
      const note = await storage.getNote(id);
      if (!note) {
        return res.status(404).json({ message: "Note not found" });
      }
      res.json(note);
    } catch (error) {
      res.status(500).json({ message: "Failed to fetch note" });
    }
  });

  app.post("/api/notes", async (req, res) => {
    try {
      const noteData = insertNoteSchema.parse(req.body);
      const note = await storage.createNote(noteData);
      
      // Generate embeddings for the new note
      await generateEmbeddings(note.id, note.content);
      
      res.json(note);
    } catch (error) {
      res.status(400).json({ message: "Invalid note data" });
    }
  });

  app.patch("/api/notes/:id", async (req, res) => {
    try {
      const id = parseInt(req.params.id);
      const updateData = updateNoteSchema.parse(req.body);
      const note = await storage.updateNote(id, updateData);
      
      if (!note) {
        return res.status(404).json({ message: "Note not found" });
      }

      // Regenerate embeddings if content changed
      if (updateData.content) {
        await storage.deleteEmbeddings(id);
        await generateEmbeddings(id, updateData.content);
      }
      
      res.json(note);
    } catch (error) {
      res.status(400).json({ message: "Invalid update data" });
    }
  });

  app.delete("/api/notes/:id", async (req, res) => {
    try {
      const id = parseInt(req.params.id);
      const deleted = await storage.deleteNote(id);
      
      if (!deleted) {
        return res.status(404).json({ message: "Note not found" });
      }
      
      res.json({ success: true });
    } catch (error) {
      res.status(500).json({ message: "Failed to delete note" });
    }
  });

  // Search endpoints
  app.get("/api/search", async (req, res) => {
    try {
      const query = req.query.q as string;
      if (!query) {
        return res.status(400).json({ message: "Query parameter required" });
      }
      
      const notes = await storage.searchNotesByText(query);
      res.json(notes);
    } catch (error) {
      res.status(500).json({ message: "Search failed" });
    }
  });

  app.post("/api/search/semantic", async (req, res) => {
    try {
      const query = req.body.query as string;
      if (!query) {
        return res.status(400).json({ message: "Query required" });
      }
      
      const results = await performSemanticSearch(query);
      res.json(results);
    } catch (error) {
      res.status(500).json({ message: "Semantic search failed" });
    }
  });

  // AI query endpoint
  app.post("/api/ai/query", async (req, res) => {
    try {
      const queryData = aiQuerySchema.parse(req.body);
      const response = await performAIQuery(queryData.query, queryData.maxResults);
      res.json(response);
    } catch (error) {
      res.status(500).json({ message: "AI query failed" });
    }
  });

  // AI summarization endpoint
  app.post("/api/ai/summarize", async (req, res) => {
    try {
      const { noteId, content, summaryType = "auto" } = req.body;
      
      if (!content || content.trim().length === 0) {
        return res.status(400).json({ error: "Content is required" });
      }
      
      // Simple fallback summary for now
      const words = content.split(/\s+/);
      const sentences = content.split(/[.!?]+/).filter(s => s.trim().length > 0);
      
      // Basic extractive summary (first sentence + a middle sentence)
      let summary = sentences[0] || "No content available.";
      if (sentences.length > 2) {
        summary += " " + sentences[Math.floor(sentences.length / 2)];
      }
      
      const result = {
        summary: summary.trim(),
        keyPoints: sentences.slice(0, 3).map(s => s.trim()).filter(s => s.length > 0),
        wordCount: words.length,
        readingTime: Math.max(1, Math.ceil(words.length / 200)),
        sentiment: "neutral",
        confidence: 0.8,
        topics: []
      };
      
      res.json(result);
    } catch (error) {
      console.error("AI summarization error:", error);
      res.status(500).json({ error: "Failed to generate summary" });
    }
  });

  // Export endpoint
  app.post("/api/export/:format", async (req, res) => {
    try {
      const format = req.params.format;
      const noteIds = req.body.noteIds as number[];
      
      if (format !== "pdf" && format !== "html") {
        return res.status(400).json({ message: "Unsupported export format" });
      }
      
      const result = await exportNotes(noteIds, format);
      res.json(result);
    } catch (error) {
      res.status(500).json({ message: "Export failed" });
    }
  });

  const httpServer = createServer(app);
  return httpServer;
}

// Helper functions for AI operations
async function generateEmbeddings(noteId: number, content: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const pythonProcess = spawn('python', [
      path.join(process.cwd(), 'server/ai_service.py'),
      'generate_embeddings',
      noteId.toString(),
      content
    ]);

    pythonProcess.on('close', (code) => {
      if (code === 0) {
        resolve();
      } else {
        reject(new Error(`Python process exited with code ${code}`));
      }
    });
  });
}

async function performAdvancedSummarization(noteId: number, content: string, summaryType: string): Promise<any> {
  return new Promise((resolve, reject) => {
    const pythonProcess = spawn('python', [
      path.join(process.cwd(), 'server/ai/summarization_engine.py'),
      'summarize',
      noteId.toString(),
      content,
      summaryType
    ]);

    let output = '';
    pythonProcess.stdout.on('data', (data) => {
      output += data.toString();
    });

    pythonProcess.on('close', (code) => {
      if (code === 0) {
        try {
          resolve(JSON.parse(output));
        } catch (e) {
          reject(new Error('Invalid JSON response from summarization service'));
        }
      } else {
        reject(new Error(`Summarization process exited with code ${code}`));
      }
    });
  });
}

async function performKnowledgeGraphAnalysis(noteId: number, content: string, title: string): Promise<any> {
  return new Promise((resolve, reject) => {
    const pythonProcess = spawn('python', [
      path.join(process.cwd(), 'server/ai/knowledge_graph.py'),
      'analyze',
      noteId.toString(),
      content,
      title
    ]);

    let output = '';
    pythonProcess.stdout.on('data', (data) => {
      output += data.toString();
    });

    pythonProcess.on('close', (code) => {
      if (code === 0) {
        try {
          resolve(JSON.parse(output));
        } catch (e) {
          reject(new Error('Invalid JSON response from knowledge graph service'));
        }
      } else {
        reject(new Error(`Knowledge graph process exited with code ${code}`));
      }
    });
  });
}

async function findRelatedNotes(noteId: number, maxResults: number): Promise<any> {
  return new Promise((resolve, reject) => {
    const pythonProcess = spawn('python', [
      path.join(process.cwd(), 'server/ai/knowledge_graph.py'),
      'find_related',
      noteId.toString(),
      maxResults.toString()
    ]);

    let output = '';
    pythonProcess.stdout.on('data', (data) => {
      output += data.toString();
    });

    pythonProcess.on('close', (code) => {
      if (code === 0) {
        try {
          resolve(JSON.parse(output));
        } catch (e) {
          reject(new Error('Invalid JSON response from knowledge graph service'));
        }
      } else {
        reject(new Error(`Related notes process exited with code ${code}`));
      }
    });
  });
}

async function suggestTags(content: string, existingTags: string[]): Promise<any> {
  return new Promise((resolve, reject) => {
    const pythonProcess = spawn('python', [
      path.join(process.cwd(), 'server/ai/knowledge_graph.py'),
      'suggest_tags',
      content,
      JSON.stringify(existingTags)
    ]);

    let output = '';
    pythonProcess.stdout.on('data', (data) => {
      output += data.toString();
    });

    pythonProcess.on('close', (code) => {
      if (code === 0) {
        try {
          resolve(JSON.parse(output));
        } catch (e) {
          reject(new Error('Invalid JSON response from tag suggestion service'));
        }
      } else {
        reject(new Error(`Tag suggestion process exited with code ${code}`));
      }
    });
  });
}

async function discoverKnowledgePatterns(): Promise<any> {
  return new Promise((resolve, reject) => {
    const pythonProcess = spawn('python', [
      path.join(process.cwd(), 'server/ai/knowledge_graph.py'),
      'discover_patterns'
    ]);

    let output = '';
    pythonProcess.stdout.on('data', (data) => {
      output += data.toString();
    });

    pythonProcess.on('close', (code) => {
      if (code === 0) {
        try {
          resolve(JSON.parse(output));
        } catch (e) {
          reject(new Error('Invalid JSON response from pattern discovery service'));
        }
      } else {
        reject(new Error(`Pattern discovery process exited with code ${code}`));
      }
    });
  });
}

async function getConceptNetwork(concept: string, depth: number): Promise<any> {
  return new Promise((resolve, reject) => {
    const pythonProcess = spawn('python', [
      path.join(process.cwd(), 'server/ai/knowledge_graph.py'),
      'concept_network',
      concept,
      depth.toString()
    ]);

    let output = '';
    pythonProcess.stdout.on('data', (data) => {
      output += data.toString();
    });

    pythonProcess.on('close', (code) => {
      if (code === 0) {
        try {
          resolve(JSON.parse(output));
        } catch (e) {
          reject(new Error('Invalid JSON response from concept network service'));
        }
      } else {
        reject(new Error(`Concept network process exited with code ${code}`));
      }
    });
  });
}

async function performSemanticSearch(query: string): Promise<any> {
  return new Promise((resolve, reject) => {
    const pythonProcess = spawn('python', [
      path.join(process.cwd(), 'server/ai_service.py'),
      'semantic_search',
      query
    ]);

    let output = '';
    pythonProcess.stdout.on('data', (data) => {
      output += data.toString();
    });

    pythonProcess.on('close', (code) => {
      if (code === 0) {
        try {
          resolve(JSON.parse(output));
        } catch (e) {
          reject(new Error('Invalid JSON response from Python service'));
        }
      } else {
        reject(new Error(`Python process exited with code ${code}`));
      }
    });
  });
}

async function performAIQuery(query: string, maxResults: number): Promise<any> {
  return new Promise((resolve, reject) => {
    const pythonProcess = spawn('python', [
      path.join(process.cwd(), 'server/ai_service.py'),
      'ai_query',
      query,
      maxResults.toString()
    ]);

    let output = '';
    pythonProcess.stdout.on('data', (data) => {
      output += data.toString();
    });

    pythonProcess.on('close', (code) => {
      if (code === 0) {
        try {
          resolve(JSON.parse(output));
        } catch (e) {
          reject(new Error('Invalid JSON response from Python service'));
        }
      } else {
        reject(new Error(`Python process exited with code ${code}`));
      }
    });
  });
}

async function exportNotes(noteIds: number[], format: string): Promise<any> {
  return new Promise((resolve, reject) => {
    const pythonProcess = spawn('python', [
      path.join(process.cwd(), 'server/ai_service.py'),
      'export_notes',
      JSON.stringify(noteIds),
      format
    ]);

    let output = '';
    pythonProcess.stdout.on('data', (data) => {
      output += data.toString();
    });

    pythonProcess.on('close', (code) => {
      if (code === 0) {
        try {
          resolve(JSON.parse(output));
        } catch (e) {
          reject(new Error('Invalid JSON response from Python service'));
        }
      } else {
        reject(new Error(`Python process exited with code ${code}`));
      }
    });
  });
}
