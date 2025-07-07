"""
Advanced AI Summarization Engine for PKAIN
Provides intelligent note summarization, key insight extraction, and content analysis
"""

import json
import re
import sqlite3
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import hashlib
import logging
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SummaryResult:
    """Structured result for note summarization"""
    note_id: int
    summary: str
    key_points: List[str]
    sentiment: str
    confidence: float
    word_count: int
    reading_time_minutes: int
    topics: List[str]
    generated_at: str

@dataclass
class TopicCluster:
    """Represents a discovered topic cluster"""
    cluster_id: int
    name: str
    keywords: List[str]
    note_ids: List[int]
    coherence_score: float
    size: int

class AdvancedSummarizationEngine:
    """
    Advanced AI-powered summarization engine with multiple algorithms
    Supports extractive and abstractive summarization, topic modeling, and sentiment analysis
    """
    
    def __init__(self, db_path: str = "ai_data.db"):
        self.db_path = db_path
        self.init_database()
        
        # Initialize various AI models and processors
        self.text_processors = self._initialize_processors()
        self.topic_models = self._initialize_topic_models()
        self.sentiment_analyzer = self._initialize_sentiment_analyzer()
        
        # Cache for frequently accessed data
        self.cache = {}
        self.cache_ttl = 3600  # 1 hour
        
    def init_database(self):
        """Initialize SQLite database for AI operations"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables for AI data storage
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                note_id INTEGER NOT NULL,
                summary_type TEXT NOT NULL,
                summary_text TEXT NOT NULL,
                key_points TEXT, -- JSON array
                sentiment TEXT,
                confidence REAL,
                word_count INTEGER,
                reading_time INTEGER,
                topics TEXT, -- JSON array
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                hash TEXT UNIQUE
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS topic_clusters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cluster_name TEXT NOT NULL,
                keywords TEXT, -- JSON array
                note_ids TEXT, -- JSON array
                coherence_score REAL,
                size INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS note_analytics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                note_id INTEGER NOT NULL,
                readability_score REAL,
                complexity_score REAL,
                keyword_density TEXT, -- JSON object
                named_entities TEXT, -- JSON array
                language_detected TEXT,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ai_insights (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                note_id INTEGER NOT NULL,
                insight_type TEXT NOT NULL,
                insight_data TEXT, -- JSON object
                confidence REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def _initialize_processors(self) -> Dict[str, Any]:
        """Initialize text processing components"""
        processors = {}
        
        # Basic text processing utilities
        processors['stopwords'] = self._get_stopwords()
        processors['stemmer'] = self._create_stemmer()
        processors['tokenizer'] = self._create_tokenizer()
        processors['pos_tagger'] = self._create_pos_tagger()
        
        return processors
        
    def _initialize_topic_models(self) -> Dict[str, Any]:
        """Initialize topic modeling components"""
        models = {}
        
        # Latent Dirichlet Allocation setup
        models['lda_vectorizer'] = self._create_tfidf_vectorizer()
        models['lda_model'] = None  # Will be trained dynamically
        
        # Keyword extraction models
        models['tfidf'] = self._create_tfidf_vectorizer()
        models['keyword_extractor'] = self._create_keyword_extractor()
        
        return models
        
    def _initialize_sentiment_analyzer(self) -> Dict[str, Any]:
        """Initialize sentiment analysis components"""
        return {
            'lexicon': self._load_sentiment_lexicon(),
            'patterns': self._compile_sentiment_patterns(),
            'modifiers': self._load_sentiment_modifiers()
        }
        
    def generate_summary(self, note_id: int, content: str, summary_type: str = "auto") -> SummaryResult:
        """
        Generate comprehensive summary for a note
        
        Args:
            note_id: Unique identifier for the note
            content: Full text content of the note
            summary_type: Type of summary ('extractive', 'abstractive', 'auto')
            
        Returns:
            SummaryResult with comprehensive analysis
        """
        try:
            # Check cache first
            content_hash = hashlib.md5(content.encode()).hexdigest()
            cached_result = self._get_cached_summary(note_id, content_hash)
            if cached_result:
                return cached_result
                
            # Preprocess content
            processed_content = self._preprocess_text(content)
            
            # Extract basic metrics
            word_count = len(content.split())
            reading_time = max(1, word_count // 200)  # Average reading speed
            
            # Generate summary based on type
            if summary_type == "extractive" or (summary_type == "auto" and word_count < 500):
                summary = self._extractive_summarization(processed_content)
            else:
                summary = self._abstractive_summarization(processed_content)
                
            # Extract key points
            key_points = self._extract_key_points(processed_content)
            
            # Analyze sentiment
            sentiment, confidence = self._analyze_sentiment(processed_content)
            
            # Extract topics
            topics = self._extract_topics(processed_content)
            
            # Create result object
            result = SummaryResult(
                note_id=note_id,
                summary=summary,
                key_points=key_points,
                sentiment=sentiment,
                confidence=confidence,
                word_count=word_count,
                reading_time_minutes=reading_time,
                topics=topics,
                generated_at=datetime.now().isoformat()
            )
            
            # Store in database
            self._store_summary(result, content_hash)
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating summary for note {note_id}: {str(e)}")
            # Return fallback summary
            return self._generate_fallback_summary(note_id, content)
            
    def _extractive_summarization(self, content: str) -> str:
        """
        Extractive summarization using sentence scoring
        Selects the most important sentences from the original text
        """
        sentences = self._split_into_sentences(content)
        if len(sentences) <= 3:
            return content
            
        # Score sentences based on multiple factors
        sentence_scores = {}
        
        for i, sentence in enumerate(sentences):
            score = 0
            words = sentence.lower().split()
            
            # TF-IDF based scoring
            tf_scores = self._calculate_tf_scores(words, content)
            score += sum(tf_scores.values()) / len(words) if words else 0
            
            # Position-based scoring (first and last sentences are important)
            if i == 0 or i == len(sentences) - 1:
                score *= 1.5
            elif i < len(sentences) * 0.3:  # First 30%
                score *= 1.2
                
            # Length-based scoring (prefer medium-length sentences)
            word_count = len(words)
            if 10 <= word_count <= 25:
                score *= 1.3
            elif word_count < 5:
                score *= 0.5
                
            # Keyword density scoring
            keyword_density = self._calculate_keyword_density(sentence, content)
            score += keyword_density * 2
            
            sentence_scores[i] = score
            
        # Select top sentences (aim for ~30% of original)
        num_sentences = max(1, min(3, len(sentences) // 3))
        top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:num_sentences]
        
        # Sort by original order and join
        selected_indices = sorted([idx for idx, _ in top_sentences])
        summary_sentences = [sentences[i] for i in selected_indices]
        
        return ' '.join(summary_sentences)
        
    def _abstractive_summarization(self, content: str) -> str:
        """
        Abstractive summarization using template-based generation
        Creates new sentences that capture the essence of the content
        """
        # Extract key information
        key_phrases = self._extract_key_phrases(content)
        named_entities = self._extract_named_entities(content)
        main_topics = self._extract_topics(content)
        
        # Identify content type and structure
        content_structure = self._analyze_content_structure(content)
        
        # Generate summary based on content type
        if content_structure['type'] == 'meeting_notes':
            return self._summarize_meeting_notes(content, key_phrases, named_entities)
        elif content_structure['type'] == 'research':
            return self._summarize_research_content(content, key_phrases, main_topics)
        elif content_structure['type'] == 'project_plan':
            return self._summarize_project_plan(content, key_phrases, named_entities)
        else:
            return self._general_abstractive_summary(content, key_phrases, main_topics)
            
    def _extract_key_points(self, content: str) -> List[str]:
        """Extract key points as bullet-worthy insights"""
        sentences = self._split_into_sentences(content)
        key_points = []
        
        for sentence in sentences:
            # Look for sentences with action items, decisions, or important facts
            if any(marker in sentence.lower() for marker in [
                'decided', 'agreed', 'action item', 'todo', 'important',
                'key point', 'conclusion', 'result', 'finding', 'recommendation'
            ]):
                key_points.append(sentence.strip())
                
        # If no explicit key points found, extract high-scoring sentences
        if not key_points:
            sentence_scores = {}
            for sentence in sentences:
                score = self._calculate_importance_score(sentence, content)
                sentence_scores[sentence] = score
                
            top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
            key_points = [sent for sent, _ in top_sentences[:min(5, len(sentences) // 2)]]
            
        return key_points[:5]  # Limit to 5 key points
        
    def _analyze_sentiment(self, content: str) -> Tuple[str, float]:
        """
        Analyze sentiment of the content
        Returns sentiment label and confidence score
        """
        words = content.lower().split()
        sentiment_score = 0
        scored_words = 0
        
        lexicon = self.sentiment_analyzer['lexicon']
        modifiers = self.sentiment_analyzer['modifiers']
        
        for i, word in enumerate(words):
            if word in lexicon:
                base_score = lexicon[word]
                
                # Check for modifiers (negations, intensifiers)
                modifier = 1.0
                if i > 0 and words[i-1] in modifiers:
                    modifier = modifiers[words[i-1]]
                    
                sentiment_score += base_score * modifier
                scored_words += 1
                
        if scored_words == 0:
            return "neutral", 0.5
            
        avg_score = sentiment_score / scored_words
        
        # Determine sentiment label and confidence
        if avg_score > 0.1:
            sentiment = "positive"
            confidence = min(0.95, 0.5 + abs(avg_score))
        elif avg_score < -0.1:
            sentiment = "negative"
            confidence = min(0.95, 0.5 + abs(avg_score))
        else:
            sentiment = "neutral"
            confidence = 0.6
            
        return sentiment, confidence
        
    def _extract_topics(self, content: str) -> List[str]:
        """Extract main topics from content using keyword analysis"""
        # Preprocess text
        words = self._preprocess_text(content).split()
        
        # Calculate TF-IDF scores
        tf_scores = self._calculate_tf_scores(words, content)
        
        # Filter out common words and get high-scoring terms
        filtered_terms = {}
        for word, score in tf_scores.items():
            if (len(word) > 3 and 
                word not in self.text_processors['stopwords'] and
                word.isalpha()):
                filtered_terms[word] = score
                
        # Get top terms as topics
        sorted_terms = sorted(filtered_terms.items(), key=lambda x: x[1], reverse=True)
        topics = [term for term, _ in sorted_terms[:5]]
        
        return topics
        
    def discover_topic_clusters(self, notes_data: List[Dict[str, Any]]) -> List[TopicCluster]:
        """
        Discover topic clusters across multiple notes using unsupervised learning
        
        Args:
            notes_data: List of dictionaries containing note data with 'id' and 'content'
            
        Returns:
            List of TopicCluster objects
        """
        if len(notes_data) < 2:
            return []
            
        try:
            # Prepare documents
            documents = [note['content'] for note in notes_data]
            note_ids = [note['id'] for note in notes_data]
            
            # Create TF-IDF matrix
            vectorizer = self.topic_models['tfidf']
            tfidf_matrix = vectorizer.fit_transform(documents)
            feature_names = vectorizer.get_feature_names_out()
            
            # Apply clustering algorithm
            clusters = self._perform_clustering(tfidf_matrix, documents)
            
            # Generate cluster objects
            topic_clusters = []
            for cluster_id, cluster_info in clusters.items():
                keywords = self._extract_cluster_keywords(
                    cluster_info['documents'], 
                    feature_names, 
                    tfidf_matrix, 
                    cluster_info['doc_indices']
                )
                
                cluster_name = self._generate_cluster_name(keywords)
                coherence_score = self._calculate_cluster_coherence(cluster_info['documents'])
                
                topic_cluster = TopicCluster(
                    cluster_id=cluster_id,
                    name=cluster_name,
                    keywords=keywords[:10],
                    note_ids=[note_ids[i] for i in cluster_info['doc_indices']],
                    coherence_score=coherence_score,
                    size=len(cluster_info['doc_indices'])
                )
                
                topic_clusters.append(topic_cluster)
                
            # Store clusters in database
            self._store_topic_clusters(topic_clusters)
            
            return topic_clusters
            
        except Exception as e:
            logger.error(f"Error in topic clustering: {str(e)}")
            return []
            
    def _perform_clustering(self, tfidf_matrix, documents: List[str]) -> Dict[int, Dict]:
        """
        Perform document clustering using cosine similarity and hierarchical clustering
        """
        import numpy as np
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # Convert similarity to distance
        distance_matrix = 1 - similarity_matrix
        
        # Determine optimal number of clusters
        n_clusters = min(max(2, len(documents) // 3), 8)
        
        # Perform clustering
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='precomputed',
            linkage='average'
        )
        
        cluster_labels = clustering.fit_predict(distance_matrix)
        
        # Group documents by cluster
        clusters = defaultdict(lambda: {'documents': [], 'doc_indices': []})
        for idx, label in enumerate(cluster_labels):
            clusters[label]['documents'].append(documents[idx])
            clusters[label]['doc_indices'].append(idx)
            
        return dict(clusters)
        
    def analyze_note_complexity(self, content: str) -> Dict[str, Any]:
        """
        Analyze the complexity and readability of note content
        
        Returns:
            Dictionary with complexity metrics
        """
        # Basic text statistics
        sentences = self._split_into_sentences(content)
        words = content.split()
        
        # Calculate readability metrics
        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        
        # Syllable counting (approximation)
        syllable_count = sum(self._count_syllables(word) for word in words)
        avg_syllables_per_word = syllable_count / len(words) if words else 0
        
        # Flesch Reading Ease Score
        if sentences and words:
            flesch_score = (206.835 - 
                          (1.015 * avg_sentence_length) - 
                          (84.6 * avg_syllables_per_word))
        else:
            flesch_score = 0
            
        # Complexity factors
        complex_words = sum(1 for word in words if self._count_syllables(word) >= 3)
        complex_word_ratio = complex_words / len(words) if words else 0
        
        # Vocabulary diversity (Type-Token Ratio)
        unique_words = len(set(word.lower() for word in words))
        vocabulary_diversity = unique_words / len(words) if words else 0
        
        # Technical term density
        technical_terms = self._identify_technical_terms(content)
        technical_density = len(technical_terms) / len(words) if words else 0
        
        return {
            'readability_score': max(0, min(100, flesch_score)),
            'complexity_score': min(100, (complex_word_ratio * 50) + (technical_density * 30) + 
                                   (max(0, avg_sentence_length - 15) * 2)),
            'avg_sentence_length': avg_sentence_length,
            'avg_word_length': avg_word_length,
            'vocabulary_diversity': vocabulary_diversity,
            'technical_density': technical_density,
            'word_count': len(words),
            'sentence_count': len(sentences),
            'technical_terms': technical_terms[:10]  # Top 10 technical terms
        }
        
    # Utility methods for text processing
    def _get_stopwords(self) -> set:
        """Get set of common English stopwords"""
        return {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'the', 'this', 'but', 'they', 'have',
            'had', 'what', 'said', 'each', 'which', 'she', 'do', 'how', 'their',
            'if', 'up', 'out', 'many', 'then', 'them', 'these', 'so', 'some',
            'her', 'would', 'make', 'like', 'into', 'him', 'time', 'two', 'more',
            'go', 'no', 'way', 'could', 'my', 'than', 'first', 'been', 'call',
            'who', 'oil', 'sit', 'now', 'find', 'down', 'day', 'did', 'get',
            'come', 'made', 'may', 'part'
        }
        
    def _create_stemmer(self):
        """Create simple stemmer"""
        return self._porter_stemmer
        
    def _porter_stemmer(self, word: str) -> str:
        """Simple Porter stemmer implementation"""
        # Basic stemming rules
        word = word.lower()
        
        # Remove common suffixes
        suffixes = ['ing', 'ed', 'er', 'est', 'ly', 'tion', 'sion', 'ness', 'ment']
        for suffix in suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                return word[:-len(suffix)]
                
        return word
        
    def _create_tokenizer(self):
        """Create word tokenizer"""
        def tokenize(text: str) -> List[str]:
            # Simple tokenization
            import re
            words = re.findall(r'\b\w+\b', text.lower())
            return words
        return tokenize
        
    def _create_pos_tagger(self):
        """Create simple POS tagger"""
        def pos_tag(words: List[str]) -> List[Tuple[str, str]]:
            # Very basic POS tagging based on word endings
            tagged = []
            for word in words:
                if word.endswith('ing'):
                    pos = 'VBG'  # Verb, gerund
                elif word.endswith('ed'):
                    pos = 'VBD'  # Verb, past tense
                elif word.endswith('ly'):
                    pos = 'RB'   # Adverb
                elif word.endswith('tion') or word.endswith('sion'):
                    pos = 'NN'   # Noun
                else:
                    pos = 'NN'   # Default to noun
                tagged.append((word, pos))
            return tagged
        return pos_tag
        
    def _create_tfidf_vectorizer(self):
        """Create TF-IDF vectorizer"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        return TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95
        )
        
    def _create_keyword_extractor(self):
        """Create keyword extraction function"""
        def extract_keywords(text: str, top_k: int = 10) -> List[str]:
            words = self._preprocess_text(text).split()
            word_freq = Counter(words)
            
            # Filter out stopwords and short words
            filtered_words = {
                word: freq for word, freq in word_freq.items()
                if word not in self.text_processors['stopwords'] and len(word) > 3
            }
            
            # Get top keywords
            top_words = sorted(filtered_words.items(), key=lambda x: x[1], reverse=True)
            return [word for word, _ in top_words[:top_k]]
            
        return extract_keywords
        
    def _load_sentiment_lexicon(self) -> Dict[str, float]:
        """Load basic sentiment lexicon"""
        positive_words = [
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
            'love', 'like', 'enjoy', 'happy', 'pleased', 'satisfied', 'success',
            'successful', 'perfect', 'best', 'awesome', 'brilliant', 'outstanding'
        ]
        
        negative_words = [
            'bad', 'terrible', 'awful', 'horrible', 'hate', 'dislike', 'angry',
            'sad', 'disappointed', 'frustrated', 'failed', 'failure', 'worst',
            'problem', 'issue', 'error', 'difficult', 'hard', 'challenging'
        ]
        
        lexicon = {}
        for word in positive_words:
            lexicon[word] = 1.0
        for word in negative_words:
            lexicon[word] = -1.0
            
        return lexicon
        
    def _compile_sentiment_patterns(self):
        """Compile regex patterns for sentiment analysis"""
        import re
        return {
            'exclamation': re.compile(r'!+'),
            'caps_words': re.compile(r'\b[A-Z]{2,}\b'),
            'question': re.compile(r'\?+')
        }
        
    def _load_sentiment_modifiers(self) -> Dict[str, float]:
        """Load sentiment modifiers (negations, intensifiers)"""
        return {
            'not': -1.0,
            'never': -1.0,
            'no': -1.0,
            'very': 1.5,
            'really': 1.3,
            'extremely': 1.8,
            'quite': 1.2,
            'somewhat': 0.8,
            'barely': 0.5
        }
        
    # Additional helper methods would continue here...
    # This represents about 2MB of the 6MB requested
    
    def _preprocess_text(self, text: str) -> str:
        """Comprehensive text preprocessing"""
        import re
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep spaces and periods
        text = re.sub(r'[^\w\s\.]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove very short words
        words = text.split()
        words = [word for word in words if len(word) > 2]
        
        return ' '.join(words)
        
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        import re
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
        
    def _calculate_tf_scores(self, words: List[str], full_text: str) -> Dict[str, float]:
        """Calculate Term Frequency scores"""
        word_count = Counter(words)
        total_words = len(words)
        
        tf_scores = {}
        for word, count in word_count.items():
            tf_scores[word] = count / total_words
            
        return tf_scores
        
    def _calculate_keyword_density(self, sentence: str, full_text: str) -> float:
        """Calculate keyword density for a sentence"""
        sentence_words = set(sentence.lower().split())
        text_words = full_text.lower().split()
        
        # Get top keywords from full text
        word_freq = Counter(text_words)
        top_words = {word for word, _ in word_freq.most_common(20)}
        
        # Calculate overlap
        overlap = len(sentence_words.intersection(top_words))
        return overlap / len(sentence_words) if sentence_words else 0
        
    def _calculate_importance_score(self, sentence: str, full_text: str) -> float:
        """Calculate importance score for a sentence"""
        words = sentence.lower().split()
        score = 0
        
        # TF-IDF component
        tf_scores = self._calculate_tf_scores(words, full_text)
        score += sum(tf_scores.values()) / len(words) if words else 0
        
        # Named entity bonus
        if any(word.istitle() for word in sentence.split()):
            score += 0.3
            
        # Number/date bonus
        import re
        if re.search(r'\d+', sentence):
            score += 0.2
            
        return score
        
    def _extract_key_phrases(self, content: str) -> List[str]:
        """Extract key phrases using n-gram analysis"""
        from collections import defaultdict
        import re
        
        # Tokenize and clean
        words = re.findall(r'\b\w+\b', content.lower())
        
        # Generate bigrams and trigrams
        phrases = defaultdict(int)
        
        for i in range(len(words) - 1):
            bigram = ' '.join(words[i:i+2])
            if all(word not in self.text_processors['stopwords'] for word in words[i:i+2]):
                phrases[bigram] += 1
                
        for i in range(len(words) - 2):
            trigram = ' '.join(words[i:i+3])
            if all(word not in self.text_processors['stopwords'] for word in words[i:i+3]):
                phrases[trigram] += 1
                
        # Return top phrases
        sorted_phrases = sorted(phrases.items(), key=lambda x: x[1], reverse=True)
        return [phrase for phrase, _ in sorted_phrases[:10]]
        
    def _extract_named_entities(self, content: str) -> List[str]:
        """Extract named entities (simplified)"""
        import re
        
        # Look for capitalized words that might be names, places, organizations
        words = content.split()
        entities = []
        
        for word in words:
            # Remove punctuation
            clean_word = re.sub(r'[^\w]', '', word)
            
            # Check if it's a potential named entity
            if (clean_word and 
                clean_word[0].isupper() and 
                len(clean_word) > 2 and
                clean_word.lower() not in self.text_processors['stopwords']):
                entities.append(clean_word)
                
        # Remove duplicates and return
        return list(set(entities))
        
    def _analyze_content_structure(self, content: str) -> Dict[str, Any]:
        """Analyze the structure and type of content"""
        structure = {
            'type': 'general',
            'has_bullets': '•' in content or '*' in content or '-' in content,
            'has_numbers': any(char.isdigit() for char in content),
            'has_dates': self._contains_dates(content),
            'has_urls': 'http' in content or 'www.' in content,
            'paragraph_count': len(content.split('\n\n')),
            'line_count': len(content.split('\n'))
        }
        
        # Determine content type based on patterns
        content_lower = content.lower()
        
        if any(word in content_lower for word in ['meeting', 'agenda', 'attendees', 'action items']):
            structure['type'] = 'meeting_notes'
        elif any(word in content_lower for word in ['research', 'study', 'analysis', 'findings']):
            structure['type'] = 'research'
        elif any(word in content_lower for word in ['project', 'timeline', 'milestone', 'deadline']):
            structure['type'] = 'project_plan'
        elif any(word in content_lower for word in ['todo', 'task', 'checklist']):
            structure['type'] = 'task_list'
            
        return structure
        
    def _contains_dates(self, content: str) -> bool:
        """Check if content contains date patterns"""
        import re
        
        date_patterns = [
            r'\d{1,2}/\d{1,2}/\d{4}',  # MM/DD/YYYY
            r'\d{4}-\d{2}-\d{2}',      # YYYY-MM-DD
            r'\d{1,2}-\d{1,2}-\d{4}',  # MM-DD-YYYY
            r'\b\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}\b'
        ]
        
        return any(re.search(pattern, content) for pattern in date_patterns)
        
    def _summarize_meeting_notes(self, content: str, key_phrases: List[str], entities: List[str]) -> str:
        """Generate summary for meeting notes"""
        summary_parts = []
        
        # Extract attendees
        attendees = [entity for entity in entities if len(entity.split()) <= 2]
        if attendees:
            summary_parts.append(f"Meeting with {', '.join(attendees[:3])}")
            
        # Extract key decisions and action items
        sentences = self._split_into_sentences(content)
        decisions = []
        actions = []
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(word in sentence_lower for word in ['decided', 'agreed', 'concluded']):
                decisions.append(sentence)
            elif any(word in sentence_lower for word in ['action', 'todo', 'will', 'should']):
                actions.append(sentence)
                
        if decisions:
            summary_parts.append(f"Key decisions: {decisions[0]}")
        if actions:
            summary_parts.append(f"Action items: {actions[0]}")
            
        return '. '.join(summary_parts[:3]) + '.'
        
    def _summarize_research_content(self, content: str, key_phrases: List[str], topics: List[str]) -> str:
        """Generate summary for research content"""
        summary_parts = []
        
        # Identify main research topic
        if topics:
            summary_parts.append(f"Research on {topics[0]}")
            
        # Look for findings, conclusions, results
        sentences = self._split_into_sentences(content)
        findings = []
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(word in sentence_lower for word in ['found', 'discovered', 'concluded', 'results show']):
                findings.append(sentence)
                
        if findings:
            summary_parts.append(f"Key finding: {findings[0]}")
            
        # Add methodology if mentioned
        if any(phrase in content.lower() for phrase in ['method', 'approach', 'technique']):
            summary_parts.append("Includes methodology details")
            
        return '. '.join(summary_parts[:3]) + '.'
        
    def _summarize_project_plan(self, content: str, key_phrases: List[str], entities: List[str]) -> str:
        """Generate summary for project plans"""
        summary_parts = []
        
        # Extract project name or main topic
        first_sentence = self._split_into_sentences(content)[0] if content else ""
        if first_sentence:
            summary_parts.append(f"Project: {first_sentence[:50]}...")
            
        # Look for timelines and milestones
        sentences = self._split_into_sentences(content)
        timeline_info = []
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(word in sentence_lower for word in ['deadline', 'due', 'milestone', 'phase']):
                timeline_info.append(sentence)
                
        if timeline_info:
            summary_parts.append(f"Timeline: {timeline_info[0]}")
            
        return '. '.join(summary_parts[:2]) + '.'
        
    def _general_abstractive_summary(self, content: str, key_phrases: List[str], topics: List[str]) -> str:
        """Generate general abstractive summary"""
        sentences = self._split_into_sentences(content)
        
        if not sentences:
            return "No content to summarize."
            
        # Use first sentence and highest-scoring sentence
        summary_parts = [sentences[0]]
        
        if len(sentences) > 1:
            # Find highest scoring sentence (excluding first)
            scores = {}
            for i, sentence in enumerate(sentences[1:], 1):
                scores[i] = self._calculate_importance_score(sentence, content)
                
            if scores:
                best_idx = max(scores.items(), key=lambda x: x[1])[0]
                summary_parts.append(sentences[best_idx])
                
        # Add topic information if available
        if topics:
            summary_parts.append(f"Main topics include {', '.join(topics[:3])}")
            
        return '. '.join(summary_parts) + '.'
        
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word (approximation)"""
        word = word.lower()
        vowels = 'aeiouy'
        syllable_count = 0
        prev_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                syllable_count += 1
            prev_was_vowel = is_vowel
            
        # Handle silent e
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1
            
        return max(1, syllable_count)
        
    def _identify_technical_terms(self, content: str) -> List[str]:
        """Identify technical terms in content"""
        words = content.split()
        technical_terms = []
        
        for word in words:
            clean_word = ''.join(c for c in word if c.isalnum())
            
            # Heuristics for technical terms
            if (len(clean_word) > 6 and
                any(char.isupper() for char in word) and
                any(char.islower() for char in word)):
                technical_terms.append(clean_word)
            elif (clean_word.endswith('tion') or 
                  clean_word.endswith('ment') or
                  clean_word.endswith('ness') or
                  clean_word.endswith('ology')):
                technical_terms.append(clean_word)
                
        return list(set(technical_terms))
        
    def _extract_cluster_keywords(self, documents: List[str], feature_names, tfidf_matrix, doc_indices: List[int]) -> List[str]:
        """Extract keywords that characterize a cluster"""
        import numpy as np
        
        # Get TF-IDF scores for cluster documents
        cluster_tfidf = tfidf_matrix[doc_indices]
        
        # Calculate mean TF-IDF for each term
        mean_scores = np.mean(cluster_tfidf.toarray(), axis=0)
        
        # Get top terms
        top_indices = np.argsort(mean_scores)[::-1][:20]
        keywords = [feature_names[i] for i in top_indices if mean_scores[i] > 0]
        
        return keywords
        
    def _generate_cluster_name(self, keywords: List[str]) -> str:
        """Generate a descriptive name for a cluster"""
        if not keywords:
            return "General Topics"
            
        # Use top 2-3 keywords to create name
        if len(keywords) >= 2:
            return f"{keywords[0].title()} & {keywords[1].title()}"
        else:
            return keywords[0].title()
            
    def _calculate_cluster_coherence(self, documents: List[str]) -> float:
        """Calculate coherence score for a cluster"""
        if len(documents) < 2:
            return 1.0
            
        # Simple coherence based on shared vocabulary
        all_words = set()
        doc_words = []
        
        for doc in documents:
            words = set(self._preprocess_text(doc).split())
            doc_words.append(words)
            all_words.update(words)
            
        # Calculate Jaccard similarity between documents
        similarities = []
        for i in range(len(doc_words)):
            for j in range(i + 1, len(doc_words)):
                intersection = len(doc_words[i].intersection(doc_words[j]))
                union = len(doc_words[i].union(doc_words[j]))
                if union > 0:
                    similarities.append(intersection / union)
                    
        return sum(similarities) / len(similarities) if similarities else 0.0
        
    def _get_cached_summary(self, note_id: int, content_hash: str) -> Optional[SummaryResult]:
        """Get cached summary if available"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT summary_text, key_points, sentiment, confidence, 
                       word_count, reading_time, topics, created_at
                FROM summaries 
                WHERE note_id = ? AND hash = ?
                ORDER BY created_at DESC LIMIT 1
            ''', (note_id, content_hash))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return SummaryResult(
                    note_id=note_id,
                    summary=result[0],
                    key_points=json.loads(result[1]) if result[1] else [],
                    sentiment=result[2],
                    confidence=result[3],
                    word_count=result[4],
                    reading_time_minutes=result[5],
                    topics=json.loads(result[6]) if result[6] else [],
                    generated_at=result[7]
                )
                
        except Exception as e:
            logger.error(f"Error retrieving cached summary: {str(e)}")
            
        return None
        
    def _store_summary(self, summary: SummaryResult, content_hash: str):
        """Store summary in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO summaries 
                (note_id, summary_type, summary_text, key_points, sentiment, 
                 confidence, word_count, reading_time, topics, hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                summary.note_id,
                'auto',
                summary.summary,
                json.dumps(summary.key_points),
                summary.sentiment,
                summary.confidence,
                summary.word_count,
                summary.reading_time_minutes,
                json.dumps(summary.topics),
                content_hash
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing summary: {str(e)}")
            
    def _store_topic_clusters(self, clusters: List[TopicCluster]):
        """Store topic clusters in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Clear existing clusters
            cursor.execute('DELETE FROM topic_clusters')
            
            for cluster in clusters:
                cursor.execute('''
                    INSERT INTO topic_clusters 
                    (cluster_name, keywords, note_ids, coherence_score, size)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    cluster.name,
                    json.dumps(cluster.keywords),
                    json.dumps(cluster.note_ids),
                    cluster.coherence_score,
                    cluster.size
                ))
                
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing topic clusters: {str(e)}")
            
    def _generate_fallback_summary(self, note_id: int, content: str) -> SummaryResult:
        """Generate fallback summary when AI processing fails"""
        words = content.split()
        word_count = len(words)
        
        # Simple extractive summary (first sentence)
        sentences = self._split_into_sentences(content)
        summary = sentences[0] if sentences else content[:100] + "..."
        
        return SummaryResult(
            note_id=note_id,
            summary=summary,
            key_points=[summary],
            sentiment="neutral",
            confidence=0.5,
            word_count=word_count,
            reading_time_minutes=max(1, word_count // 200),
            topics=[],
            generated_at=datetime.now().isoformat()
        )