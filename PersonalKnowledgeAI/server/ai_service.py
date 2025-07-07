#!/usr/bin/env python3
import sys
import json
import os
import sqlite3
from typing import List, Dict, Any, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from pathlib import Path
import markdown
import weasyprint
from datetime import datetime

# Initialize the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Database path
DB_PATH = 'notes.db'
EMBEDDINGS_INDEX_PATH = 'embeddings.index'

class AIService:
    def __init__(self):
        self.model = model
        self.embedding_dim = 384  # all-MiniLM-L6-v2 dimension
        self.index = None
        self.note_chunks = []  # Store chunk metadata
        self.init_database()
        self.load_or_create_index()
    
    def init_database(self):
        """Initialize SQLite database for storing note metadata"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS notes (
                id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                tags TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY,
                note_id INTEGER,
                embedding TEXT NOT NULL,
                chunk_text TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                FOREIGN KEY (note_id) REFERENCES notes (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def load_or_create_index(self):
        """Load existing FAISS index or create a new one"""
        if os.path.exists(EMBEDDINGS_INDEX_PATH):
            self.index = faiss.read_index(EMBEDDINGS_INDEX_PATH)
            self.load_chunk_metadata()
        else:
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            self.note_chunks = []
    
    def load_chunk_metadata(self):
        """Load chunk metadata from database"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT note_id, chunk_text, chunk_index 
            FROM embeddings 
            ORDER BY id
        ''')
        self.note_chunks = [
            {'note_id': row[0], 'chunk_text': row[1], 'chunk_index': row[2]}
            for row in cursor.fetchall()
        ]
        conn.close()
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence or word boundary
            if end < len(text):
                last_period = chunk.rfind('.')
                last_space = chunk.rfind(' ')
                if last_period > chunk_size * 0.7:
                    chunk = chunk[:last_period + 1]
                elif last_space > chunk_size * 0.7:
                    chunk = chunk[:last_space]
            
            chunks.append(chunk.strip())
            start = start + len(chunk) - overlap
            
            if start >= len(text):
                break
        
        return chunks
    
    def generate_embeddings(self, note_id: int, content: str):
        """Generate and store embeddings for a note"""
        chunks = self.chunk_text(content)
        embeddings = self.model.encode(chunks)
        
        # Store in database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            cursor.execute('''
                INSERT INTO embeddings (note_id, embedding, chunk_text, chunk_index)
                VALUES (?, ?, ?, ?)
            ''', (note_id, json.dumps(embedding.tolist()), chunk, i))
            
            # Add to FAISS index
            self.index.add(embedding.reshape(1, -1))
            self.note_chunks.append({
                'note_id': note_id,
                'chunk_text': chunk,
                'chunk_index': i
            })
        
        conn.commit()
        conn.close()
        
        # Save FAISS index
        faiss.write_index(self.index, EMBEDDINGS_INDEX_PATH)
    
    def semantic_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Perform semantic search using embeddings"""
        if self.index.ntotal == 0:
            return []
        
        query_embedding = self.model.encode([query])
        
        # Search FAISS index
        scores, indices = self.index.search(query_embedding, min(k, self.index.ntotal))
        
        results = []
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        for score, idx in zip(scores[0], indices[0]):
            if idx >= len(self.note_chunks):
                continue
                
            chunk_info = self.note_chunks[idx]
            
            # Get note details
            cursor.execute('''
                SELECT id, title, content, tags, created_at, updated_at
                FROM notes WHERE id = ?
            ''', (chunk_info['note_id'],))
            
            note_row = cursor.fetchone()
            if note_row:
                results.append({
                    'note': {
                        'id': note_row[0],
                        'title': note_row[1],
                        'content': note_row[2],
                        'tags': json.loads(note_row[3]),
                        'createdAt': note_row[4],
                        'updatedAt': note_row[5]
                    },
                    'score': float(score),
                    'matchedText': chunk_info['chunk_text']
                })
        
        conn.close()
        return results
    
    def ai_query(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """Answer questions based on the knowledge base"""
        # First, get relevant context through semantic search
        search_results = self.semantic_search(query, max_results)
        
        if not search_results:
            return {
                'answer': "I couldn't find any relevant information in your notes to answer this question.",
                'sources': [],
                'confidence': 0.0
            }
        
        # Prepare context from search results
        context_chunks = []
        for result in search_results:
            context_chunks.append(f"From '{result['note']['title']}': {result['matchedText']}")
        
        context = "\n\n".join(context_chunks)
        
        # Simple extractive answer generation (can be enhanced with LLM)
        answer = self.generate_extractive_answer(query, context, search_results)
        
        return {
            'answer': answer,
            'sources': search_results,
            'confidence': search_results[0]['score'] if search_results else 0.0
        }
    
    def generate_extractive_answer(self, query: str, context: str, sources: List[Dict]) -> str:
        """Generate an extractive answer from the context"""
        # Simple keyword-based extraction
        query_words = set(query.lower().split())
        
        best_chunk = ""
        best_score = 0
        
        for source in sources:
            chunk = source['matchedText']
            chunk_words = set(chunk.lower().split())
            overlap = len(query_words.intersection(chunk_words))
            
            if overlap > best_score:
                best_score = overlap
                best_chunk = chunk
        
        if best_chunk:
            # Try to extract a relevant sentence
            sentences = best_chunk.split('.')
            for sentence in sentences:
                sentence_words = set(sentence.lower().split())
                if len(query_words.intersection(sentence_words)) >= min(2, len(query_words)):
                    return sentence.strip() + "."
        
        return best_chunk if best_chunk else "No relevant answer found."
    
    def export_notes(self, note_ids: List[int], format: str) -> Dict[str, str]:
        """Export notes in specified format"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        if note_ids:
            placeholders = ','.join('?' * len(note_ids))
            cursor.execute(f'''
                SELECT title, content, tags, created_at
                FROM notes WHERE id IN ({placeholders})
                ORDER BY created_at DESC
            ''', note_ids)
        else:
            cursor.execute('''
                SELECT title, content, tags, created_at
                FROM notes ORDER BY created_at DESC
            ''')
        
        notes = cursor.fetchall()
        conn.close()
        
        if format == 'html':
            return self.export_to_html(notes)
        elif format == 'pdf':
            return self.export_to_pdf(notes)
        else:
            return {'error': 'Unsupported format'}
    
    def export_to_html(self, notes: List[Tuple]) -> Dict[str, str]:
        """Export notes to HTML"""
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>PKAIN Notes Export</title>
            <style>
                body { font-family: 'Inter', sans-serif; margin: 40px; line-height: 1.6; }
                .note { margin-bottom: 40px; border-bottom: 1px solid #eee; padding-bottom: 20px; }
                .note-title { font-size: 24px; font-weight: bold; margin-bottom: 10px; }
                .note-meta { color: #666; font-size: 14px; margin-bottom: 20px; }
                .note-content { font-size: 16px; }
                .tag { background: #e7f3ff; padding: 4px 8px; border-radius: 4px; font-size: 12px; margin-right: 8px; }
            </style>
        </head>
        <body>
            <h1>PKAIN Notes Export</h1>
            <p>Generated on """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
        """
        
        for note in notes:
            title, content, tags_json, created_at = note
            tags = json.loads(tags_json) if tags_json else []
            
            html_content += f"""
            <div class="note">
                <div class="note-title">{title}</div>
                <div class="note-meta">
                    Created: {created_at}
                    {' | Tags: ' + ' '.join(f'<span class="tag">{tag}</span>' for tag in tags) if tags else ''}
                </div>
                <div class="note-content">{markdown.markdown(content)}</div>
            </div>
            """
        
        html_content += """
        </body>
        </html>
        """
        
        filename = f"pkain_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return {'filename': filename, 'path': os.path.abspath(filename)}
    
    def export_to_pdf(self, notes: List[Tuple]) -> Dict[str, str]:
        """Export notes to PDF"""
        html_result = self.export_to_html(notes)
        html_path = html_result['path']
        
        pdf_filename = html_result['filename'].replace('.html', '.pdf')
        
        try:
            weasyprint.HTML(filename=html_path).write_pdf(pdf_filename)
            return {'filename': pdf_filename, 'path': os.path.abspath(pdf_filename)}
        except Exception as e:
            return {'error': f'PDF generation failed: {str(e)}'}

def main():
    if len(sys.argv) < 2:
        print(json.dumps({'error': 'No command specified'}))
        return
    
    command = sys.argv[1]
    ai_service = AIService()
    
    try:
        if command == 'generate_embeddings':
            note_id = int(sys.argv[2])
            content = sys.argv[3]
            ai_service.generate_embeddings(note_id, content)
            print(json.dumps({'success': True}))
        
        elif command == 'semantic_search':
            query = sys.argv[2]
            results = ai_service.semantic_search(query)
            print(json.dumps(results))
        
        elif command == 'ai_query':
            query = sys.argv[2]
            max_results = int(sys.argv[3]) if len(sys.argv) > 3 else 5
            response = ai_service.ai_query(query, max_results)
            print(json.dumps(response))
        
        elif command == 'export_notes':
            note_ids = json.loads(sys.argv[2])
            format = sys.argv[3]
            result = ai_service.export_notes(note_ids, format)
            print(json.dumps(result))
        
        else:
            print(json.dumps({'error': 'Unknown command'}))
    
    except Exception as e:
        print(json.dumps({'error': str(e)}))

if __name__ == '__main__':
    main()
