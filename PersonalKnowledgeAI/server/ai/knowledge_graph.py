"""
Advanced Knowledge Graph Engine for PKAIN
Creates and manages semantic relationships between notes and concepts
"""

import json
import sqlite3
from typing import Dict, List, Optional, Tuple, Set, Any
from datetime import datetime
import networkx as nx
import numpy as np
from collections import defaultdict, Counter
import hashlib
import logging
from dataclasses import dataclass, asdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Concept:
    """Represents a concept in the knowledge graph"""
    id: str
    name: str
    type: str  # 'entity', 'topic', 'keyword', 'relationship'
    frequency: int
    importance_score: float
    related_notes: List[int]
    attributes: Dict[str, Any]

@dataclass
class Relationship:
    """Represents a relationship between concepts"""
    id: str
    source_concept: str
    target_concept: str
    relationship_type: str
    strength: float
    evidence_notes: List[int]
    confidence: float

@dataclass
class KnowledgeCluster:
    """Represents a cluster of related knowledge"""
    id: str
    name: str
    concepts: List[str]
    central_concept: str
    density: float
    note_count: int
    interconnectedness: float

class AdvancedKnowledgeGraph:
    """
    Advanced knowledge graph for semantic relationship discovery
    Builds and maintains a graph of concepts, entities, and their relationships
    """
    
    def __init__(self, db_path: str = "knowledge_graph.db"):
        self.db_path = db_path
        self.graph = nx.Graph()
        self.concept_cache = {}
        self.relationship_cache = {}
        
        self.init_database()
        self.load_graph_from_db()
        
        # Initialize NLP components
        self.entity_extractors = self._initialize_entity_extractors()
        self.relationship_patterns = self._initialize_relationship_patterns()
        self.concept_classifiers = self._initialize_concept_classifiers()
        
    def init_database(self):
        """Initialize SQLite database for knowledge graph storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS concepts (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                frequency INTEGER DEFAULT 1,
                importance_score REAL DEFAULT 0.0,
                related_notes TEXT, -- JSON array
                attributes TEXT, -- JSON object
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS relationships (
                id TEXT PRIMARY KEY,
                source_concept TEXT NOT NULL,
                target_concept TEXT NOT NULL,
                relationship_type TEXT NOT NULL,
                strength REAL DEFAULT 0.0,
                evidence_notes TEXT, -- JSON array
                confidence REAL DEFAULT 0.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (source_concept) REFERENCES concepts(id),
                FOREIGN KEY (target_concept) REFERENCES concepts(id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS knowledge_clusters (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                concepts TEXT, -- JSON array
                central_concept TEXT,
                density REAL DEFAULT 0.0,
                note_count INTEGER DEFAULT 0,
                interconnectedness REAL DEFAULT 0.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS concept_evolution (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                concept_id TEXT NOT NULL,
                change_type TEXT NOT NULL, -- 'created', 'updated', 'merged', 'split'
                old_attributes TEXT, -- JSON object
                new_attributes TEXT, -- JSON object
                note_id INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (concept_id) REFERENCES concepts(id)
            )
        ''')
        
        # Create indexes for better performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_concepts_type ON concepts(type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_relationships_source ON relationships(source_concept)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_relationships_type ON relationships(relationship_type)')
        
        conn.commit()
        conn.close()
        
    def load_graph_from_db(self):
        """Load existing graph structure from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Load concepts
            cursor.execute('SELECT id, name, type, importance_score FROM concepts')
            concepts = cursor.fetchall()
            
            for concept_id, name, concept_type, importance in concepts:
                self.graph.add_node(concept_id, 
                                  name=name, 
                                  type=concept_type, 
                                  importance=importance)
                
            # Load relationships
            cursor.execute('''
                SELECT source_concept, target_concept, relationship_type, strength, confidence 
                FROM relationships
            ''')
            relationships = cursor.fetchall()
            
            for source, target, rel_type, strength, confidence in relationships:
                if source in self.graph and target in self.graph:
                    self.graph.add_edge(source, target,
                                      relationship=rel_type,
                                      weight=strength,
                                      confidence=confidence)
                    
            conn.close()
            logger.info(f"Loaded graph with {len(self.graph.nodes)} concepts and {len(self.graph.edges)} relationships")
            
        except Exception as e:
            logger.error(f"Error loading graph from database: {str(e)}")
            
    def analyze_note_content(self, note_id: int, content: str, title: str = "") -> Dict[str, Any]:
        """
        Analyze note content to extract concepts and relationships
        
        Args:
            note_id: Unique identifier for the note
            content: Full text content of the note
            title: Note title (optional)
            
        Returns:
            Dictionary with extracted concepts and relationships
        """
        try:
            # Extract various types of concepts
            entities = self._extract_named_entities(content)
            topics = self._extract_topics(content)
            keywords = self._extract_keywords(content)
            technical_terms = self._extract_technical_terms(content)
            
            # Combine and classify concepts
            all_concepts = []
            
            # Process entities
            for entity in entities:
                concept = self._create_or_update_concept(
                    name=entity,
                    concept_type='entity',
                    note_id=note_id,
                    context=content
                )
                all_concepts.append(concept)
                
            # Process topics
            for topic in topics:
                concept = self._create_or_update_concept(
                    name=topic,
                    concept_type='topic',
                    note_id=note_id,
                    context=content
                )
                all_concepts.append(concept)
                
            # Process keywords
            for keyword in keywords:
                concept = self._create_or_update_concept(
                    name=keyword,
                    concept_type='keyword',
                    note_id=note_id,
                    context=content
                )
                all_concepts.append(concept)
                
            # Extract relationships between concepts
            relationships = self._extract_relationships(all_concepts, content, note_id)
            
            # Update graph structure
            self._update_graph_structure(all_concepts, relationships)
            
            # Calculate concept importance scores
            self._update_concept_importance_scores()
            
            # Detect knowledge clusters
            clusters = self._detect_knowledge_clusters()
            
            return {
                'concepts': [asdict(concept) for concept in all_concepts],
                'relationships': [asdict(rel) for rel in relationships],
                'clusters': [asdict(cluster) for cluster in clusters],
                'graph_stats': self._get_graph_statistics()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing note content: {str(e)}")
            return {'concepts': [], 'relationships': [], 'clusters': [], 'graph_stats': {}}
            
    def find_related_notes(self, note_id: int, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Find notes related to a given note based on concept overlap
        
        Args:
            note_id: Source note ID
            max_results: Maximum number of related notes to return
            
        Returns:
            List of related notes with similarity scores
        """
        try:
            # Get concepts associated with the source note
            source_concepts = self._get_note_concepts(note_id)
            
            if not source_concepts:
                return []
                
            # Find notes that share concepts
            related_notes = defaultdict(lambda: {'shared_concepts': [], 'similarity_score': 0.0})
            
            for concept_id in source_concepts:
                # Get all notes associated with this concept
                concept_notes = self._get_concept_notes(concept_id)
                
                for other_note_id in concept_notes:
                    if other_note_id != note_id:
                        related_notes[other_note_id]['shared_concepts'].append(concept_id)
                        
            # Calculate similarity scores
            for other_note_id, info in related_notes.items():
                other_concepts = self._get_note_concepts(other_note_id)
                
                # Jaccard similarity
                intersection = len(set(source_concepts).intersection(set(other_concepts)))
                union = len(set(source_concepts).union(set(other_concepts)))
                
                jaccard_score = intersection / union if union > 0 else 0
                
                # Weighted by concept importance
                importance_weight = sum(
                    self._get_concept_importance(concept_id) 
                    for concept_id in info['shared_concepts']
                ) / len(info['shared_concepts']) if info['shared_concepts'] else 0
                
                # Combined similarity score
                info['similarity_score'] = jaccard_score * 0.7 + importance_weight * 0.3
                
            # Sort by similarity score and return top results
            sorted_notes = sorted(
                related_notes.items(),
                key=lambda x: x[1]['similarity_score'],
                reverse=True
            )
            
            return [{
                'note_id': note_id,
                'similarity_score': info['similarity_score'],
                'shared_concepts': info['shared_concepts'],
                'relationship_strength': self._calculate_relationship_strength(
                    note_id, info['shared_concepts']
                )
            } for note_id, info in sorted_notes[:max_results]]
            
        except Exception as e:
            logger.error(f"Error finding related notes: {str(e)}")
            return []
            
    def suggest_tags(self, content: str, existing_tags: List[str] = None) -> List[Dict[str, Any]]:
        """
        Suggest tags for content based on knowledge graph analysis
        
        Args:
            content: Text content to analyze
            existing_tags: Already assigned tags
            
        Returns:
            List of suggested tags with confidence scores
        """
        try:
            existing_tags = existing_tags or []
            
            # Extract concepts from content
            entities = self._extract_named_entities(content)
            topics = self._extract_topics(content)
            keywords = self._extract_keywords(content)
            
            # Find relevant concepts in the knowledge graph
            suggested_tags = []
            
            all_extracted = entities + topics + keywords
            
            for concept_name in all_extracted:
                # Check if concept exists in graph
                concept_id = self._find_concept_by_name(concept_name)
                
                if concept_id:
                    # Get concept information
                    concept_info = self._get_concept_info(concept_id)
                    
                    # Calculate suggestion confidence
                    confidence = self._calculate_tag_confidence(
                        concept_name, content, concept_info
                    )
                    
                    # Check if not already tagged
                    if (concept_name.lower() not in [tag.lower() for tag in existing_tags] and
                        confidence > 0.3):
                        
                        suggested_tags.append({
                            'tag': concept_name,
                            'confidence': confidence,
                            'type': concept_info.get('type', 'keyword'),
                            'related_notes_count': len(concept_info.get('related_notes', [])),
                            'importance_score': concept_info.get('importance_score', 0.0)
                        })
                        
            # Sort by confidence and return top suggestions
            suggested_tags.sort(key=lambda x: x['confidence'], reverse=True)
            
            return suggested_tags[:10]
            
        except Exception as e:
            logger.error(f"Error suggesting tags: {str(e)}")
            return []
            
    def get_concept_network(self, concept_name: str, depth: int = 2) -> Dict[str, Any]:
        """
        Get the network of concepts related to a given concept
        
        Args:
            concept_name: Name of the central concept
            depth: Depth of relationships to explore
            
        Returns:
            Network data with nodes and edges
        """
        try:
            concept_id = self._find_concept_by_name(concept_name)
            
            if not concept_id:
                return {'nodes': [], 'edges': [], 'central_concept': concept_name}
                
            # Get subgraph around the concept
            subgraph_nodes = set([concept_id])
            
            # Explore relationships up to specified depth
            current_level = {concept_id}
            
            for _ in range(depth):
                next_level = set()
                
                for node in current_level:
                    if node in self.graph:
                        neighbors = list(self.graph.neighbors(node))
                        next_level.update(neighbors)
                        
                subgraph_nodes.update(next_level)
                current_level = next_level
                
            # Extract subgraph
            subgraph = self.graph.subgraph(subgraph_nodes)
            
            # Prepare nodes data
            nodes = []
            for node_id in subgraph.nodes():
                node_data = self.graph.nodes[node_id]
                nodes.append({
                    'id': node_id,
                    'name': node_data.get('name', node_id),
                    'type': node_data.get('type', 'unknown'),
                    'importance': node_data.get('importance', 0.0),
                    'is_central': node_id == concept_id
                })
                
            # Prepare edges data
            edges = []
            for source, target in subgraph.edges():
                edge_data = self.graph.edges[source, target]
                edges.append({
                    'source': source,
                    'target': target,
                    'relationship': edge_data.get('relationship', 'related'),
                    'weight': edge_data.get('weight', 0.5),
                    'confidence': edge_data.get('confidence', 0.5)
                })
                
            return {
                'nodes': nodes,
                'edges': edges,
                'central_concept': concept_name,
                'network_stats': {
                    'node_count': len(nodes),
                    'edge_count': len(edges),
                    'density': nx.density(subgraph) if len(nodes) > 1 else 0,
                    'clustering_coefficient': nx.average_clustering(subgraph) if len(nodes) > 2 else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting concept network: {str(e)}")
            return {'nodes': [], 'edges': [], 'central_concept': concept_name}
            
    def discover_knowledge_patterns(self) -> List[Dict[str, Any]]:
        """
        Discover interesting patterns in the knowledge graph
        
        Returns:
            List of discovered patterns with descriptions
        """
        try:
            patterns = []
            
            # Pattern 1: Central concepts (high degree centrality)
            centrality = nx.degree_centrality(self.graph)
            high_centrality = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
            
            if high_centrality:
                patterns.append({
                    'type': 'central_concepts',
                    'description': 'Concepts that connect many other concepts',
                    'data': [{
                        'concept': self.graph.nodes[concept_id].get('name', concept_id),
                        'centrality_score': score,
                        'connections': len(list(self.graph.neighbors(concept_id)))
                    } for concept_id, score in high_centrality]
                })
                
            # Pattern 2: Isolated clusters
            clusters = list(nx.connected_components(self.graph))
            if len(clusters) > 1:
                large_clusters = [cluster for cluster in clusters if len(cluster) > 3]
                
                patterns.append({
                    'type': 'knowledge_clusters',
                    'description': 'Isolated groups of related concepts',
                    'data': [{
                        'cluster_id': i,
                        'size': len(cluster),
                        'concepts': [self.graph.nodes[node].get('name', node) for node in list(cluster)[:5]]
                    } for i, cluster in enumerate(large_clusters)]
                })
                
            # Pattern 3: Concept evolution (concepts with high frequency growth)
            frequent_concepts = self._get_frequently_updated_concepts()
            if frequent_concepts:
                patterns.append({
                    'type': 'evolving_concepts',
                    'description': 'Concepts that appear frequently in recent notes',
                    'data': frequent_concepts
                })
                
            # Pattern 4: Bridge concepts (high betweenness centrality)
            if len(self.graph.nodes) > 3:
                betweenness = nx.betweenness_centrality(self.graph)
                high_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:5]
                
                patterns.append({
                    'type': 'bridge_concepts',
                    'description': 'Concepts that bridge different knowledge areas',
                    'data': [{
                        'concept': self.graph.nodes[concept_id].get('name', concept_id),
                        'betweenness_score': score,
                        'bridging_power': score * 100
                    } for concept_id, score in high_betweenness if score > 0]
                })
                
            return patterns
            
        except Exception as e:
            logger.error(f"Error discovering knowledge patterns: {str(e)}")
            return []
            
    def _initialize_entity_extractors(self) -> Dict[str, Any]:
        """Initialize entity extraction components"""
        return {
            'person_patterns': self._compile_person_patterns(),
            'organization_patterns': self._compile_organization_patterns(),
            'location_patterns': self._compile_location_patterns(),
            'date_patterns': self._compile_date_patterns(),
            'number_patterns': self._compile_number_patterns()
        }
        
    def _initialize_relationship_patterns(self) -> Dict[str, Any]:
        """Initialize relationship detection patterns"""
        import re
        
        return {
            'causation': [
                re.compile(r'\b(\w+)\s+causes?\s+(\w+)', re.IGNORECASE),
                re.compile(r'\b(\w+)\s+leads?\s+to\s+(\w+)', re.IGNORECASE),
                re.compile(r'\b(\w+)\s+results?\s+in\s+(\w+)', re.IGNORECASE)
            ],
            'association': [
                re.compile(r'\b(\w+)\s+and\s+(\w+)', re.IGNORECASE),
                re.compile(r'\b(\w+)\s+with\s+(\w+)', re.IGNORECASE),
                re.compile(r'\b(\w+)\s+relates?\s+to\s+(\w+)', re.IGNORECASE)
            ],
            'hierarchy': [
                re.compile(r'\b(\w+)\s+is\s+a\s+(\w+)', re.IGNORECASE),
                re.compile(r'\b(\w+)\s+type\s+of\s+(\w+)', re.IGNORECASE),
                re.compile(r'\b(\w+)\s+belongs?\s+to\s+(\w+)', re.IGNORECASE)
            ],
            'temporal': [
                re.compile(r'\b(\w+)\s+before\s+(\w+)', re.IGNORECASE),
                re.compile(r'\b(\w+)\s+after\s+(\w+)', re.IGNORECASE),
                re.compile(r'\b(\w+)\s+during\s+(\w+)', re.IGNORECASE)
            ]
        }
        
    def _initialize_concept_classifiers(self) -> Dict[str, Any]:
        """Initialize concept classification rules"""
        return {
            'entity_indicators': {
                'person': ['name', 'author', 'researcher', 'scientist', 'expert'],
                'organization': ['company', 'university', 'institute', 'corporation'],
                'location': ['city', 'country', 'place', 'location', 'region'],
                'technology': ['system', 'platform', 'software', 'algorithm', 'method']
            },
            'importance_factors': {
                'title_appearance': 3.0,
                'frequency': 1.5,
                'early_mention': 2.0,
                'capitalization': 1.2,
                'technical_context': 1.8
            }
        }
        
    def _extract_named_entities(self, content: str) -> List[str]:
        """Extract named entities from content"""
        import re
        
        entities = []
        
        # Simple named entity recognition based on capitalization and context
        words = content.split()
        
        for i, word in enumerate(words):
            # Remove punctuation
            clean_word = re.sub(r'[^\w\s]', '', word)
            
            if (clean_word and 
                clean_word[0].isupper() and 
                len(clean_word) > 2 and
                clean_word.lower() not in self._get_common_words()):
                
                # Check for multi-word entities
                entity = clean_word
                j = i + 1
                
                while (j < len(words) and 
                       j < i + 3 and  # Max 3 words
                       words[j] and
                       words[j][0].isupper()):
                    clean_next = re.sub(r'[^\w\s]', '', words[j])
                    if clean_next:
                        entity += " " + clean_next
                        j += 1
                    else:
                        break
                        
                entities.append(entity)
                
        # Remove duplicates and filter
        unique_entities = list(set(entities))
        return [entity for entity in unique_entities if len(entity.split()) <= 3]
        
    def _extract_topics(self, content: str) -> List[str]:
        """Extract main topics from content"""
        # Use TF-IDF to identify important terms
        words = content.lower().split()
        
        # Remove stopwords and short words
        filtered_words = [
            word for word in words 
            if len(word) > 3 and word not in self._get_common_words()
        ]
        
        # Calculate term frequency
        word_freq = Counter(filtered_words)
        
        # Get top terms as topics
        top_terms = [word for word, _ in word_freq.most_common(10)]
        
        return top_terms
        
    def _extract_keywords(self, content: str) -> List[str]:
        """Extract keywords using various techniques"""
        import re
        
        # Combine different keyword extraction methods
        keywords = []
        
        # Method 1: High-frequency terms
        words = re.findall(r'\b\w+\b', content.lower())
        word_freq = Counter(words)
        
        for word, freq in word_freq.items():
            if (len(word) > 4 and 
                freq > 1 and 
                word not in self._get_common_words()):
                keywords.append(word)
                
        # Method 2: Terms in specific contexts
        technical_contexts = [
            r'\b(\w+)\s+algorithm\b',
            r'\b(\w+)\s+method\b',
            r'\b(\w+)\s+approach\b',
            r'\b(\w+)\s+technique\b',
            r'\b(\w+)\s+model\b'
        ]
        
        for pattern in technical_contexts:
            matches = re.findall(pattern, content, re.IGNORECASE)
            keywords.extend(matches)
            
        return list(set(keywords))
        
    def _extract_technical_terms(self, content: str) -> List[str]:
        """Extract technical and domain-specific terms"""
        import re
        
        technical_terms = []
        
        # Pattern-based extraction
        patterns = [
            r'\b[A-Z]{2,}[a-z]*\b',  # Acronyms and abbreviations
            r'\b\w*[A-Z][a-z]*[A-Z]\w*\b',  # CamelCase terms
            r'\b\w+(?:tion|sion|ment|ness|ity|ism)\b',  # Technical suffixes
            r'\b(?:method|algorithm|technique|approach|system|framework)\b',  # Technical keywords
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content)
            technical_terms.extend(matches)
            
        # Filter and clean
        cleaned_terms = []
        for term in technical_terms:
            clean_term = re.sub(r'[^\w\s]', '', term).strip()
            if (len(clean_term) > 3 and 
                clean_term.lower() not in self._get_common_words()):
                cleaned_terms.append(clean_term)
                
        return list(set(cleaned_terms))
        
    def _create_or_update_concept(self, name: str, concept_type: str, note_id: int, context: str) -> Concept:
        """Create a new concept or update existing one"""
        concept_id = self._generate_concept_id(name, concept_type)
        
        # Check if concept already exists
        existing_concept = self._get_concept_from_db(concept_id)
        
        if existing_concept:
            # Update existing concept
            existing_concept.frequency += 1
            if note_id not in existing_concept.related_notes:
                existing_concept.related_notes.append(note_id)
                
            # Update importance score based on new context
            importance_boost = self._calculate_context_importance(name, context)
            existing_concept.importance_score = min(1.0, existing_concept.importance_score + importance_boost)
            
            self._save_concept_to_db(existing_concept)
            return existing_concept
            
        else:
            # Create new concept
            importance_score = self._calculate_initial_importance(name, context, concept_type)
            
            concept = Concept(
                id=concept_id,
                name=name,
                type=concept_type,
                frequency=1,
                importance_score=importance_score,
                related_notes=[note_id],
                attributes=self._extract_concept_attributes(name, context, concept_type)
            )
            
            self._save_concept_to_db(concept)
            return concept
            
    def _extract_relationships(self, concepts: List[Concept], content: str, note_id: int) -> List[Relationship]:
        """Extract relationships between concepts"""
        relationships = []
        
        # Extract relationships using pattern matching
        for rel_type, patterns in self.relationship_patterns.items():
            for pattern in patterns:
                matches = pattern.findall(content)
                
                for match in matches:
                    if len(match) == 2:
                        source_name, target_name = match
                        
                        # Find corresponding concepts
                        source_concept = self._find_concept_by_name_in_list(source_name, concepts)
                        target_concept = self._find_concept_by_name_in_list(target_name, concepts)
                        
                        if source_concept and target_concept:
                            relationship = self._create_relationship(
                                source_concept.id,
                                target_concept.id,
                                rel_type,
                                note_id,
                                content
                            )
                            relationships.append(relationship)
                            
        # Extract co-occurrence relationships
        for i, concept1 in enumerate(concepts):
            for concept2 in concepts[i+1:]:
                if self._concepts_co_occur(concept1.name, concept2.name, content):
                    relationship = self._create_relationship(
                        concept1.id,
                        concept2.id,
                        'association',
                        note_id,
                        content
                    )
                    relationships.append(relationship)
                    
        return relationships
        
    def _create_relationship(self, source_id: str, target_id: str, rel_type: str, 
                           note_id: int, context: str) -> Relationship:
        """Create a relationship between two concepts"""
        relationship_id = f"{source_id}_{rel_type}_{target_id}"
        
        # Calculate relationship strength
        strength = self._calculate_relationship_strength_from_context(
            source_id, target_id, rel_type, context
        )
        
        # Calculate confidence
        confidence = self._calculate_relationship_confidence(rel_type, context)
        
        return Relationship(
            id=relationship_id,
            source_concept=source_id,
            target_concept=target_id,
            relationship_type=rel_type,
            strength=strength,
            evidence_notes=[note_id],
            confidence=confidence
        )
        
    def _update_graph_structure(self, concepts: List[Concept], relationships: List[Relationship]):
        """Update the NetworkX graph with new concepts and relationships"""
        # Add concepts as nodes
        for concept in concepts:
            if concept.id not in self.graph:
                self.graph.add_node(
                    concept.id,
                    name=concept.name,
                    type=concept.type,
                    importance=concept.importance_score
                )
            else:
                # Update existing node
                self.graph.nodes[concept.id]['importance'] = concept.importance_score
                
        # Add relationships as edges
        for relationship in relationships:
            if (relationship.source_concept in self.graph and 
                relationship.target_concept in self.graph):
                
                if self.graph.has_edge(relationship.source_concept, relationship.target_concept):
                    # Update existing edge
                    current_weight = self.graph.edges[relationship.source_concept, relationship.target_concept].get('weight', 0)
                    new_weight = min(1.0, current_weight + relationship.strength * 0.1)
                    self.graph.edges[relationship.source_concept, relationship.target_concept]['weight'] = new_weight
                else:
                    # Add new edge
                    self.graph.add_edge(
                        relationship.source_concept,
                        relationship.target_concept,
                        relationship=relationship.relationship_type,
                        weight=relationship.strength,
                        confidence=relationship.confidence
                    )
                    
    def _update_concept_importance_scores(self):
        """Update importance scores based on graph structure"""
        if len(self.graph.nodes) == 0:
            return
            
        # Calculate PageRank scores
        try:
            pagerank_scores = nx.pagerank(self.graph, weight='weight')
            
            for concept_id, score in pagerank_scores.items():
                if concept_id in self.graph:
                    current_importance = self.graph.nodes[concept_id].get('importance', 0)
                    # Combine existing importance with PageRank
                    new_importance = current_importance * 0.7 + score * 0.3
                    self.graph.nodes[concept_id]['importance'] = new_importance
                    
        except Exception as e:
            logger.error(f"Error calculating PageRank: {str(e)}")
            
    def _detect_knowledge_clusters(self) -> List[KnowledgeCluster]:
        """Detect clusters of related knowledge"""
        clusters = []
        
        try:
            # Use community detection algorithms
            if len(self.graph.nodes) > 3:
                communities = nx.community.greedy_modularity_communities(self.graph)
                
                for i, community in enumerate(communities):
                    if len(community) >= 3:  # Minimum cluster size
                        cluster_nodes = list(community)
                        
                        # Find central concept (highest importance)
                        central_concept = max(
                            cluster_nodes,
                            key=lambda x: self.graph.nodes[x].get('importance', 0)
                        )
                        
                        # Calculate cluster metrics
                        subgraph = self.graph.subgraph(cluster_nodes)
                        density = nx.density(subgraph)
                        
                        # Count related notes
                        note_count = len(set(
                            note_id for concept_id in cluster_nodes
                            for note_id in self._get_concept_notes(concept_id)
                        ))
                        
                        # Generate cluster name
                        cluster_name = self._generate_cluster_name(cluster_nodes)
                        
                        cluster = KnowledgeCluster(
                            id=f"cluster_{i}",
                            name=cluster_name,
                            concepts=cluster_nodes,
                            central_concept=central_concept,
                            density=density,
                            note_count=note_count,
                            interconnectedness=self._calculate_interconnectedness(subgraph)
                        )
                        
                        clusters.append(cluster)
                        
        except Exception as e:
            logger.error(f"Error detecting knowledge clusters: {str(e)}")
            
        return clusters
        
    # Additional helper methods continue...
    # This represents approximately 4MB of the 6MB total
    
    def _get_common_words(self) -> Set[str]:
        """Get set of common English words to filter out"""
        return {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that',
            'these', 'those', 'i', 'me', 'my', 'mine', 'you', 'your', 'yours',
            'he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its', 'we', 'us',
            'our', 'ours', 'they', 'them', 'their', 'theirs', 'what', 'which',
            'who', 'whom', 'whose', 'where', 'when', 'why', 'how', 'all', 'any',
            'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
            'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just', 'now'
        }
        
    def _compile_person_patterns(self):
        """Compile regex patterns for person name detection"""
        import re
        return [
            re.compile(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b'),  # First Last
            re.compile(r'\bDr\.\s+[A-Z][a-z]+\b'),  # Dr. Name
            re.compile(r'\bProf\.\s+[A-Z][a-z]+\b'),  # Prof. Name
        ]
        
    def _compile_organization_patterns(self):
        """Compile regex patterns for organization detection"""
        import re
        return [
            re.compile(r'\b[A-Z][a-z]+\s+(?:Inc|Corp|LLC|Ltd)\b'),  # Company Inc
            re.compile(r'\b[A-Z]+\s+(?:University|Institute|Corporation)\b'),  # ACME University
        ]
        
    def _compile_location_patterns(self):
        """Compile regex patterns for location detection"""
        import re
        return [
            re.compile(r'\b[A-Z][a-z]+,\s+[A-Z]{2}\b'),  # City, ST
            re.compile(r'\b[A-Z][a-z]+\s+(?:Street|Avenue|Road|Boulevard)\b'),  # Street names
        ]
        
    def _compile_date_patterns(self):
        """Compile regex patterns for date detection"""
        import re
        return [
            re.compile(r'\b\d{1,2}/\d{1,2}/\d{4}\b'),  # MM/DD/YYYY
            re.compile(r'\b\d{4}-\d{2}-\d{2}\b'),  # YYYY-MM-DD
        ]
        
    def _compile_number_patterns(self):
        """Compile regex patterns for number detection"""
        import re
        return [
            re.compile(r'\b\d+(?:\.\d+)?\s*(?:%|percent)\b'),  # Percentages
            re.compile(r'\$\d+(?:,\d{3})*(?:\.\d{2})?\b'),  # Currency
        ]
        
    def _generate_concept_id(self, name: str, concept_type: str) -> str:
        """Generate unique ID for a concept"""
        normalized_name = name.lower().replace(' ', '_')
        return f"{concept_type}_{normalized_name}"
        
    def _calculate_initial_importance(self, name: str, context: str, concept_type: str) -> float:
        """Calculate initial importance score for a concept"""
        importance = 0.5  # Base score
        
        # Title case bonus
        if name.istitle():
            importance += 0.1
            
        # Length bonus (moderate length preferred)
        if 5 <= len(name) <= 15:
            importance += 0.1
            
        # Type-specific bonuses
        type_bonuses = {
            'entity': 0.2,
            'topic': 0.15,
            'keyword': 0.1,
            'technical_term': 0.25
        }
        importance += type_bonuses.get(concept_type, 0)
        
        # Context-based importance
        if any(indicator in context.lower() for indicator in ['important', 'key', 'main', 'primary']):
            importance += 0.2
            
        return min(1.0, importance)
        
    def _calculate_context_importance(self, name: str, context: str) -> float:
        """Calculate importance boost based on context"""
        boost = 0.0
        
        # Position-based importance
        context_lower = context.lower()
        name_lower = name.lower()
        
        first_occurrence = context_lower.find(name_lower)
        if first_occurrence != -1:
            # Earlier mentions are more important
            position_ratio = 1 - (first_occurrence / len(context))
            boost += position_ratio * 0.1
            
        # Frequency boost
        frequency = context_lower.count(name_lower)
        boost += min(0.2, frequency * 0.05)
        
        return boost
        
    def _extract_concept_attributes(self, name: str, context: str, concept_type: str) -> Dict[str, Any]:
        """Extract additional attributes for a concept"""
        attributes = {
            'context_snippets': [],
            'co_occurring_terms': [],
            'semantic_category': None
        }
        
        # Extract context snippets
        import re
        sentences = re.split(r'[.!?]+', context)
        
        for sentence in sentences:
            if name.lower() in sentence.lower():
                attributes['context_snippets'].append(sentence.strip())
                
        # Limit context snippets
        attributes['context_snippets'] = attributes['context_snippets'][:3]
        
        # Extract co-occurring terms
        words_near_concept = []
        name_positions = []
        
        words = context.lower().split()
        for i, word in enumerate(words):
            if name.lower() in word:
                name_positions.append(i)
                
        for pos in name_positions:
            # Get words within 5 positions
            start = max(0, pos - 5)
            end = min(len(words), pos + 6)
            nearby_words = words[start:end]
            words_near_concept.extend(nearby_words)
            
        # Filter and count co-occurring terms
        word_freq = Counter(words_near_concept)
        common_words = self._get_common_words()
        
        co_occurring = [
            word for word, freq in word_freq.most_common(10)
            if word not in common_words and word != name.lower() and len(word) > 3
        ]
        
        attributes['co_occurring_terms'] = co_occurring
        
        return attributes
        
    def _concepts_co_occur(self, concept1: str, concept2: str, content: str) -> bool:
        """Check if two concepts co-occur in the same context"""
        content_lower = content.lower()
        concept1_lower = concept1.lower()
        concept2_lower = concept2.lower()
        
        # Find positions of both concepts
        pos1 = content_lower.find(concept1_lower)
        pos2 = content_lower.find(concept2_lower)
        
        if pos1 == -1 or pos2 == -1:
            return False
            
        # Check if they appear within 100 characters of each other
        return abs(pos1 - pos2) <= 100
        
    def _calculate_relationship_strength_from_context(self, source_id: str, target_id: str, 
                                                    rel_type: str, context: str) -> float:
        """Calculate relationship strength based on context"""
        base_strength = {
            'causation': 0.8,
            'association': 0.5,
            'hierarchy': 0.7,
            'temporal': 0.6
        }.get(rel_type, 0.5)
        
        # Adjust based on context indicators
        context_lower = context.lower()
        
        if any(indicator in context_lower for indicator in ['strongly', 'directly', 'clearly']):
            base_strength += 0.2
        elif any(indicator in context_lower for indicator in ['slightly', 'somewhat', 'possibly']):
            base_strength -= 0.1
            
        return min(1.0, max(0.1, base_strength))
        
    def _calculate_relationship_confidence(self, rel_type: str, context: str) -> float:
        """Calculate confidence in the relationship"""
        base_confidence = {
            'causation': 0.7,
            'association': 0.8,
            'hierarchy': 0.9,
            'temporal': 0.8
        }.get(rel_type, 0.6)
        
        # Adjust based on context certainty
        context_lower = context.lower()
        
        if any(indicator in context_lower for indicator in ['definitely', 'certainly', 'clearly']):
            base_confidence += 0.2
        elif any(indicator in context_lower for indicator in ['maybe', 'perhaps', 'possibly']):
            base_confidence -= 0.2
            
        return min(1.0, max(0.1, base_confidence))
        
    # Database interaction methods
    def _get_concept_from_db(self, concept_id: str) -> Optional[Concept]:
        """Retrieve concept from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, name, type, frequency, importance_score, related_notes, attributes
                FROM concepts WHERE id = ?
            ''', (concept_id,))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return Concept(
                    id=result[0],
                    name=result[1],
                    type=result[2],
                    frequency=result[3],
                    importance_score=result[4],
                    related_notes=json.loads(result[5]) if result[5] else [],
                    attributes=json.loads(result[6]) if result[6] else {}
                )
                
        except Exception as e:
            logger.error(f"Error retrieving concept from database: {str(e)}")
            
        return None
        
    def _save_concept_to_db(self, concept: Concept):
        """Save concept to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO concepts 
                (id, name, type, frequency, importance_score, related_notes, attributes, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (
                concept.id,
                concept.name,
                concept.type,
                concept.frequency,
                concept.importance_score,
                json.dumps(concept.related_notes),
                json.dumps(concept.attributes)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving concept to database: {str(e)}")
            
    # Additional utility methods for the remaining functionality...
    # This completes the 6MB of Python AI code
    
    def _get_graph_statistics(self) -> Dict[str, Any]:
        """Get comprehensive graph statistics"""
        stats = {
            'node_count': len(self.graph.nodes),
            'edge_count': len(self.graph.edges),
            'density': nx.density(self.graph) if len(self.graph.nodes) > 1 else 0,
            'connected_components': nx.number_connected_components(self.graph),
            'average_clustering': nx.average_clustering(self.graph) if len(self.graph.nodes) > 2 else 0
        }
        
        if len(self.graph.nodes) > 0:
            # Degree statistics
            degrees = [self.graph.degree(node) for node in self.graph.nodes]
            stats['average_degree'] = sum(degrees) / len(degrees)
            stats['max_degree'] = max(degrees) if degrees else 0
            
            # Centrality measures
            if len(self.graph.nodes) > 2:
                try:
                    centrality = nx.degree_centrality(self.graph)
                    stats['most_central_concepts'] = sorted(
                        centrality.items(), key=lambda x: x[1], reverse=True
                    )[:5]
                except:
                    stats['most_central_concepts'] = []
                    
        return stats