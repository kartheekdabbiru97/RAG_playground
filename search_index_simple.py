#!/usr/bin/env python3
"""
Simple Local FAISS Index Search Script

This script loads a saved FAISS index and allows you to search for similar embeddings
using local sentence transformers. No API calls required!
"""

import os
import pandas as pd
import numpy as np
import faiss
import pickle
from typing import List, Tuple, Optional
from pathlib import Path
import logging
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleFAISSSearcher:
    """Class to handle FAISS index loading and searching with local sentence transformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the simple FAISS searcher.
        
        Args:
            model_name: Sentence transformer model name
        """
        self.model_name = model_name
        self.index = None
        self.data_df = None
        self.index_info = None
        
        # Load sentence transformer model
        logger.info(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        logger.info(f"Model loaded successfully")
    
    def load_index(self, index_dir: str, base_name: str = "embeddings"):
        """
        Load FAISS index and associated data.
        
        Args:
            index_dir: Directory containing the index files
            base_name: Base name of the index files
        """
        index_path = Path(index_dir)
        
        # Load FAISS index
        faiss_path = index_path / f"{base_name}.faiss"
        if not faiss_path.exists():
            raise FileNotFoundError(f"FAISS index not found at: {faiss_path}")
        
        self.index = faiss.read_index(str(faiss_path))
        logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")
        
        # Load index info
        info_path = index_path / f"{base_name}_info.pkl"
        if info_path.exists():
            with open(info_path, 'rb') as f:
                self.index_info = pickle.load(f)
            logger.info(f"Index info: {self.index_info}")
        
        # Load data DataFrame
        data_path = index_path / f"{base_name}_data.csv"
        if data_path.exists():
            self.data_df = pd.read_csv(data_path)
            logger.info(f"Loaded data with {len(self.data_df)} rows")
        else:
            logger.warning("Data CSV not found. Only index search will be available.")
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for a single text using sentence transformers.
        
        Args:
            text: Text string to embed
            
        Returns:
            numpy array of the embedding
        """
        try:
            # Generate embedding using sentence transformers
            embedding = self.model.encode([text], convert_to_numpy=True)
            return embedding.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    def search(self, query: str, k: int = 5) -> List[Tuple[int, float, Optional[str]]]:
        """
        Search for similar embeddings.
        
        Args:
            query: Query text to search for
            k: Number of results to return
            
        Returns:
            List of tuples (index, similarity_score, original_text)
        """
        if self.index is None:
            raise ValueError("Index not loaded. Call load_index() first.")
        
        # Get embedding for query
        query_embedding = self.get_embedding(query)
        
        # Search index
        scores, indices = self.index.search(query_embedding, k)
        
        # Prepare results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            original_text = None
            if self.data_df is not None and idx < len(self.data_df):
                # Get the original text from the data
                original_text = self.data_df.iloc[idx].to_dict()
            
            results.append((int(idx), float(score), original_text))
        
        return results


def main():
    """Main function to run interactive search."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Search FAISS index for similar embeddings using local sentence transformers")
    parser.add_argument("index_dir", help="Directory containing the FAISS index files")
    parser.add_argument("--base-name", default="embeddings", help="Base name of the index files")
    parser.add_argument("--query", help="Query text to search for (if not provided, will run interactively)")
    parser.add_argument("--k", type=int, default=5, help="Number of results to return")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", 
                       help="Sentence transformer model name")
    
    args = parser.parse_args()
    
    try:
        # Initialize searcher
        searcher = SimpleFAISSSearcher(model_name=args.model)
        
        # Load index
        searcher.load_index(args.index_dir, args.base_name)
        
        if args.query:
            # Single query mode
            results = searcher.search(args.query, args.k)
            print(f"\nSearch results for: '{args.query}'")
            print("=" * 50)
            
            for i, (idx, score, data) in enumerate(results, 1):
                print(f"\n{i}. Score: {score:.4f}")
                if data:
                    for key, value in data.items():
                        print(f"   {key}: {value}")
                else:
                    print(f"   Index: {idx}")
        
        else:
            # Interactive mode
            print("Local FAISS Index Search - Interactive Mode")
            print("Enter queries to search (type 'quit' to exit)")
            print("=" * 50)
            
            while True:
                query = input("\nEnter search query: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not query:
                    continue
                
                try:
                    results = searcher.search(query, args.k)
                    print(f"\nSearch results for: '{query}'")
                    print("-" * 30)
                    
                    for i, (idx, score, data) in enumerate(results, 1):
                        print(f"\n{i}. Score: {score:.4f}")
                        if data:
                            for key, value in data.items():
                                print(f"   {key}: {value}")
                        else:
                            print(f"   Index: {idx}")
                            
                except Exception as e:
                    print(f"Error during search: {e}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == "__main__":
    main() 