# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Knowledge Graph construction from text using word-level tokenization and SVO extraction.
"""

import spacy
import networkx as nx
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict, Counter
import numpy as np
from tqdm import tqdm

class KnowledgeGraphConstructor:
    """
    Constructs a Knowledge Graph from text by extracting SVO triplets from sentences.
    """
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Initialize the KG constructor.
        
        Args:
            model_name: Name of the spaCy model to use for dependency parsing
        """
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            print(f"Downloading spaCy model: {model_name}")
            import spacy.cli
            spacy.cli.download(model_name)
            self.nlp = spacy.load(model_name)
        
        self.graph = nx.MultiDiGraph()
        self.sentence_to_triplets = defaultdict(list)
        self.entity_to_sentences = defaultdict(set)
    
    def extract_svo_triplets(self, sentence: str) -> List[Tuple[str, str, str]]:
        """
        Extract subject-verb-object triplets from a sentence.
        
        Args:
            sentence: Input sentence text
            
        Returns:
            List of (subject, relation, object) triplets
        """
        doc = self.nlp(sentence)
        triplets = []
        
        for sent in doc.sents:
            sent_text = sent.text
            
            for token in sent:
                if token.dep_ in ("ROOT", "VERB"):
                    # Get the subject
                    subjects = [tok for tok in token.lefts if tok.dep_ in ("nsubj", "nsubjpass")]
                    if not subjects:  # Imperative or other cases
                        subjects = [tok for tok in token.children if tok.dep_ in ("nsubj", "nsubjpass")]
                    
                    # Get the object
                    objects = [tok for tok in token.rights if tok.dep_ in ("dobj", "pobj")]
                    if not objects:  # Check for other object types
                        objects = [tok for tok in token.children if tok.dep_ in ("dobj", "pobj", "attr")]
                    
                    # Add triplets for valid SVO patterns
                    for subj in subjects:
                        for obj in objects:
                            # Get the full noun phrase for subject and object
                            subj_text = ' '.join([t.text for t in subj.subtree])
                            obj_text = ' '.join([t.text for t in obj.subtree])
                            relation = token.lemma_
                            
                            if subj_text and obj_text and relation:
                                triplets.append((subj_text.lower(), relation, obj_text.lower()))
        
        return triplets
    
    def add_document(self, text: str, doc_id: str = None) -> None:
        """
        Process a document and add its triplets to the knowledge graph.
        
        Args:
            text: Input text (can be multiple sentences)
            doc_id: Optional document ID for reference
        """
        doc = self.nlp(text)
        
        for sent_idx, sent in enumerate(doc.sents):
            sent_text = sent.text.strip()
            if not sent_text:
                continue
                
            # Extract SVO triplets
            triplets = self.extract_svo_triplets(sent_text)
            
            # Add to sentence tracking
            sent_id = f"{doc_id}_s{sent_idx}" if doc_id else f"s{sent_idx}"
            self.sentence_to_triplets[sent_id] = triplets
            
            # Add to graph
            for subj, rel, obj in triplets:
                # Add nodes if they don't exist
                if not self.graph.has_node(subj):
                    self.graph.add_node(subj, type='entity', mentions=0)
                if not self.graph.has_node(obj):
                    self.graph.add_node(obj, type='entity', mentions=0)
                
                # Add edge with relation
                self.graph.add_edge(subj, obj, relation=rel, sentence_id=sent_id)
                
                # Update entity mentions
                self.graph.nodes[subj]['mentions'] += 1
                self.graph.nodes[obj]['mentions'] += 1
                
                # Track which sentences mention each entity
                self.entity_to_sentences[subj].add(sent_id)
                self.entity_to_sentences[obj].add(sent_id)
    
    def get_sentence_entities(self, sentence_id: str) -> Set[str]:
        """Get all entities mentioned in a sentence."""
        entities = set()
        for subj, _, obj in self.sentence_to_triplets.get(sentence_id, []):
            entities.add(subj)
            entities.add(obj)
        return entities
    
    def get_entity_sentences(self, entity: str) -> Set[str]:
        """Get all sentence IDs where an entity is mentioned."""
        return self.entity_to_sentences.get(entity, set())
    
    def get_connected_entities(self, entity: str, max_depth: int = 2) -> Set[str]:
        """
        Get all entities connected to the given entity within max_depth hops.
        
        Args:
            entity: The starting entity
            max_depth: Maximum number of hops to traverse
            
        Returns:
            Set of connected entity names
        """
        if entity not in self.graph:
            return set()
            
        visited = {entity}
        queue = [(entity, 0)]
        
        while queue:
            current, depth = queue.pop(0)
            
            if depth >= max_depth:
                continue
                
            for neighbor in self.graph.neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))
        
        return visited - {entity}  # Exclude the starting entity
    
    def save_graph(self, filepath: str) -> None:
        """Save the knowledge graph to a file."""
        import json
        from networkx.readwrite import json_graph
        
        data = {
            'graph': json_graph.node_link_data(self.graph),
            'sentence_to_triplets': {k: list(v) for k, v in self.sentence_to_triplets.items()},
            'entity_to_sentences': {k: list(v) for k, v in self.entity_to_sentences.items()}
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load_graph(cls, filepath: str, model_name: str = "en_core_web_sm"):
        """Load a knowledge graph from a file."""
        import json
        from networkx.readwrite import json_graph
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        kg = cls(model_name)
        kg.graph = json_graph.node_link_graph(data['graph'])
        kg.sentence_to_triplets = {k: [tuple(t) for t in v] 
                                 for k, v in data['sentence_to_triplets'].items()}
        kg.entity_to_sentences = {k: set(v) for k, v in data['entity_to_sentences'].items()}
        
        return kg
