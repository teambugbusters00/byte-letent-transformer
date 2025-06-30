# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Knowledge Graph (KG) processing utilities for extracting entities and relationships
from preprocessed text.
"""
import itertools
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Set, Any

import networkx as nx
import spacy
from spacy.tokens import Doc, Span

class KGProcessor:
    """
    Process text to extract entities and relationships for knowledge graph construction.
    Uses spaCy for NLP tasks like NER and dependency parsing.
    """
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Initialize the KG processor with a spaCy language model.
        
        Args:
            model_name: Name of the spaCy language model to use
        """
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            # If model is not found, download it
            import subprocess
            import sys
            print(f"Downloading spaCy model: {model_name}")
            subprocess.check_call([sys.executable, "-m", "spacy", "download", model_name])
            self.nlp = spacy.load(model_name)
        
        # Define entity types to keep (customize based on your needs)
        self.keep_entity_types = {
            'PERSON', 'NORP', 'FAC', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT',
            'WORK_OF_ART', 'LAW', 'LANGUAGE', 'DATE', 'TIME', 'PERCENT', 'MONEY',
            'QUANTITY', 'ORDINAL', 'CARDINAL'
        }
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract named entities from text.
        
        Args:
            text: Input text to process
            
        Returns:
            List of entities with their types and positions
        """
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            if ent.label_ in self.keep_entity_types:
                entities.append({
                    'text': ent.text,
                    'type': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char
                })
        
        return entities
    
    def extract_noun_chunks(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract noun chunks from text.
        
        Args:
            text: Input text to process
            
        Returns:
            List of noun chunks with their positions
        """
        doc = self.nlp(text)
        chunks = []
        
        for chunk in doc.noun_chunks:
            chunks.append({
                'text': chunk.text,
                'root_text': chunk.root.text,
                'root_dep': chunk.root.dep_,
                'start': chunk.start_char,
                'end': chunk.end_char
            })
        
        return chunks
    
    def extract_relations(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract subject-verb-object (SVO) triplets from text.
        
        Args:
            text: Input text to process
            
        Returns:
            List of SVO triplets
        """
        doc = self.nlp(text)
        relations = []
        
        for sent in doc.sents:
            sent_relations = self._extract_svo(sent)
            relations.extend(sent_relations)
        
        return relations
    
    def _extract_svo(self, sent: Span) -> List[Dict[str, Any]]:
        """Extract SVO triplets from a single sentence."""
        subj = []
        obj = []
        verb = None
        
        for token in sent:
            if "subj" in token.dep_:
                subj.append(token.text)
            elif "obj" in token.dep_:
                obj.append(token.text)
            elif token.pos_ == "VERB":
                verb = token.lemma_
        
        if subj and verb and obj:
            return [{
                'subject': " ".join(subj),
                'relation': verb,
                'object': " ".join(obj),
                'sentence': sent.text
            }]
        return []
    
    def build_knowledge_graph(
        self, 
        text: str,
        include_entities: bool = True,
        include_noun_chunks: bool = True,
        include_relations: bool = True
    ) -> nx.Graph:
        """
        Build a knowledge graph from text.
        
        Args:
            text: Input text to process
            include_entities: Whether to include named entities in the graph
            include_noun_chunks: Whether to include noun chunks in the graph
            include_relations: Whether to include SVO relations in the graph
            
        Returns:
            A NetworkX graph representing the knowledge graph
        """
        G = nx.Graph()
        
        # Add nodes and edges based on selected components
        if include_entities:
            entities = self.extract_entities(text)
            for entity in entities:
                G.add_node(entity['text'], type='entity', entity_type=entity['type'])
        
        if include_noun_chunks:
            chunks = self.extract_noun_chunks(text)
            for chunk in chunks:
                G.add_node(chunk['text'], type='chunk', root=chunk['root_text'], dep=chunk['root_dep'])
        
        if include_relations:
            relations = self.extract_relations(text)
            for rel in relations:
                if rel['subject'] and rel['object']:
                    G.add_edge(
                        rel['subject'], 
                        rel['object'], 
                        relation=rel['relation'],
                        sentence=rel['sentence']
                    )
        
        return G
    
    def process_document(
        self, 
        text: str,
        clean_text: bool = True,
        cleaner: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Process a document and return extracted information.
        
        Args:
            text: Input text to process
            clean_text: Whether to clean the text before processing
            cleaner: Optional TextCleaner instance to use for cleaning
            
        Returns:
            Dictionary containing extracted information
        """
        if clean_text:
            if cleaner is None:
                from .text_cleaner import create_default_cleaner
                cleaner = create_default_cleaner()
            text = cleaner.clean_text(text)
        
        return {
            'text': text,
            'entities': self.extract_entities(text),
            'noun_chunks': self.extract_noun_chunks(text),
            'relations': self.extract_relations(text),
            'knowledge_graph': self.build_knowledge_graph(text)
        }
