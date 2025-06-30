# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Document processing pipeline for text cleaning and knowledge graph construction.
"""
import json
import networkx as nx
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .text_cleaner import TextCleaner, create_default_cleaner
from .kg_processor import KGProcessor


class DocumentProcessor:
    """
    A pipeline for processing documents through text cleaning and knowledge graph construction.
    """
    
    def __init__(
        self,
        cleaner: Optional[TextCleaner] = None,
        kg_processor: Optional[KGProcessor] = None,
        output_dir: Optional[Union[str, Path]] = None
    ):
        """
        Initialize the document processor.
        
        Args:
            cleaner: Optional TextCleaner instance (uses default if None)
            kg_processor: Optional KGProcessor instance (creates one if None)
            output_dir: Optional directory to save processed outputs
        """
        self.cleaner = cleaner if cleaner is not None else create_default_cleaner()
        self.kg_processor = kg_processor if kg_processor is not None else KGProcessor()
        self.output_dir = Path(output_dir) if output_dir else None
        
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def process_text(
        self,
        text: str,
        clean_text: bool = True,
        extract_kg: bool = True,
        save_output: bool = False,
        output_prefix: str = "doc"
    ) -> Dict[str, Any]:
        """
        Process a single text document.
        
        Args:
            text: Input text to process
            clean_text: Whether to clean the text before processing
            extract_kg: Whether to extract knowledge graph information
            save_output: Whether to save the output to disk
            output_prefix: Prefix for output files
            
        Returns:
            Dictionary containing processed information
        """
        result = {'original_text': text}
        
        # Clean text if requested
        if clean_text:
            cleaned_text = self.cleaner.clean_text(text)
            result['cleaned_text'] = cleaned_text
        else:
            cleaned_text = text
        
        # Extract knowledge graph information if requested
        if extract_kg:
            kg_result = self.kg_processor.process_document(
                cleaned_text,
                clean_text=False,  # Already cleaned if needed
                cleaner=self.cleaner if clean_text else None
            )
            result.update(kg_result)
        
        # Save results if requested
        if save_output and self.output_dir:
            self._save_result(result, output_prefix)
        
        return result
    
    def process_file(
        self,
        input_file: Union[str, Path],
        output_file: Optional[Union[str, Path]] = None,
        **process_kwargs
    ) -> Dict[str, Any]:
        """
        Process a text file.
        
        Args:
            input_file: Path to input text file
            output_file: Optional path to output file (uses input file name if None)
            **process_kwargs: Additional arguments to pass to process_text()
            
        Returns:
            Dictionary containing processed information
        """
        input_file = Path(input_file)
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        if output_file is None and self.output_dir:
            output_file = self.output_dir / f"{input_file.stem}_processed.json"
        
        output_prefix = output_file.stem if output_file else input_file.stem
        result = self.process_text(text, output_prefix=output_prefix, **process_kwargs)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
        
        return result
    
    def _save_result(self, result: Dict[str, Any], prefix: str) -> None:
        """
        Save processing results to disk.
        
        Args:
            result: Processing results to save
            prefix: Prefix for output files
        """
        if not self.output_dir:
            return
            
        # Create a copy of the result to avoid modifying the original
        result_copy = result.copy()
        
        # Save cleaned text
        if 'cleaned_text' in result_copy:
            with open(self.output_dir / f"{prefix}_cleaned.txt", 'w', encoding='utf-8') as f:
                f.write(result_copy['cleaned_text'])
        
        # Save entities
        if 'entities' in result_copy:
            with open(self.output_dir / f"{prefix}_entities.json", 'w', encoding='utf-8') as f:
                json.dump(result_copy['entities'], f, indent=2, ensure_ascii=False)
        
        # Save relations
        if 'relations' in result_copy:
            with open(self.output_dir / f"{prefix}_relations.json", 'w', encoding='utf-8') as f:
                json.dump(result_copy['relations'], f, indent=2, ensure_ascii=False)
        
        # Handle knowledge graph serialization
        if 'knowledge_graph' in result_copy:
            kg = result_copy.pop('knowledge_graph')
            if kg is not None:
                # Convert graph to node-link format for serialization
                kg_data = nx.node_link_data(kg)
                with open(self.output_dir / f"{prefix}_kg.json", 'w', encoding='utf-8') as f:
                    json.dump(kg_data, f, indent=2, ensure_ascii=False)
        
        # Save remaining result data
        with open(self.output_dir / f"{prefix}_full.json", 'w', encoding='utf-8') as f:
            json.dump(result_copy, f, indent=2, ensure_ascii=False)


def create_default_pipeline(output_dir: Optional[Union[str, Path]] = None) -> DocumentProcessor:
    """
    Create a document processor with default settings.
    
    Args:
        output_dir: Optional directory to save processed outputs
        
    Returns:
        Configured DocumentProcessor instance
    """
    cleaner = create_default_cleaner()
    kg_processor = KGProcessor()
    return DocumentProcessor(cleaner=cleaner, kg_processor=kg_processor, output_dir=output_dir)
