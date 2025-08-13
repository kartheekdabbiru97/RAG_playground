#!/usr/bin/env python3
"""
Simple Local CSV Embedding and FAISS Indexing Script

This script uses sentence-transformers for local embedding generation.
No API calls required - everything runs locally!
"""

import os
import pandas as pd
import numpy as np
import faiss
import pickle
from typing import List, Optional, Tuple
from pathlib import Path
import logging
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleCSVEmbedder:
    """Class to handle CSV loading, local embedding generation, and FAISS indexing."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the simple CSV embedder.
        
        Args:
            model_name: Sentence transformer model name
        """
        self.model_name = model_name
        
        # Load sentence transformer model
        logger.info(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        logger.info(f"Model loaded successfully")
    
    def get_embeddings(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        """
        Get embeddings for a list of texts using sentence transformers.
        
        Args:
            texts: List of text strings to embed
            batch_size: Number of texts to process in each batch
            
        Returns:
            numpy array of embeddings
        """
        logger.info(f"Generating embeddings for {len(texts)} texts...")
        
        try:
            # Generate embeddings using sentence transformers
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            
            logger.info(f"Generated embeddings with shape: {embeddings.shape}")
            return embeddings.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def load_csv(self, csv_path: str, text_column: str, max_rows: Optional[int] = None) -> pd.DataFrame:
        """
        Load CSV file and prepare for embedding.
        
        Args:
            csv_path: Path to the CSV file
            text_column: Name of the column containing text to embed
            max_rows: Maximum number of rows to process (for testing)
            
        Returns:
            DataFrame with the loaded data
        """
        logger.info(f"Loading CSV from: {csv_path}")
        
        # Load CSV
        df = pd.read_csv(csv_path)
        
        # Validate text column exists
        if text_column not in df.columns:
            raise ValueError(f"Text column '{text_column}' not found in CSV. Available columns: {list(df.columns)}")
        
        # Limit rows if specified
        if max_rows:
            df = df.head(max_rows)
            logger.info(f"Limited to {max_rows} rows for processing")
        
        # Remove rows with empty text
        initial_count = len(df)
        df = df.dropna(subset=[text_column])
        df = df[df[text_column].str.strip() != '']
        final_count = len(df)
        
        if initial_count != final_count:
            logger.info(f"Removed {initial_count - final_count} rows with empty text")
        
        logger.info(f"Loaded {len(df)} rows from CSV")
        return df
    
    def create_faiss_index(self, embeddings: np.ndarray, index_type: str = "flat") -> faiss.Index:
        """
        Create a FAISS index from embeddings.
        
        Args:
            embeddings: numpy array of embeddings
            index_type: Type of FAISS index ("flat", "ivf", "hnsw")
            
        Returns:
            FAISS index
        """
        dimension = embeddings.shape[1]
        
        if index_type == "flat":
            # Simple flat index - exact search, good for small datasets
            index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        elif index_type == "ivf":
            # Inverted file index - approximate search, good for large datasets
            nlist = min(4096, max(1, embeddings.shape[0] // 30))  # Number of clusters
            quantizer = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
            # Train the index
            index.train(embeddings)
        elif index_type == "hnsw":
            # Hierarchical Navigable Small World - good balance of speed and accuracy
            index = faiss.IndexHNSWFlat(dimension, 32)  # 32 neighbors
            index.hnsw.efConstruction = 200
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
        
        # Add embeddings to index
        index.add(embeddings)
        
        logger.info(f"Created FAISS index of type '{index_type}' with {index.ntotal} vectors")
        return index
    
    def save_index(self, index: faiss.Index, output_dir: str, base_name: str):
        """
        Save FAISS index and metadata.
        
        Args:
            index: FAISS index to save
            output_dir: Directory to save the index
            base_name: Base name for the output files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_path = output_path / f"{base_name}.faiss"
        faiss.write_index(index, str(index_path))
        logger.info(f"Saved FAISS index to: {index_path}")
        
        # Save index info
        info = {
            "ntotal": index.ntotal,
            "dimension": index.d,
            "index_type": type(index).__name__,
            "model_name": self.model_name
        }
        
        info_path = output_path / f"{base_name}_info.pkl"
        with open(info_path, 'wb') as f:
            pickle.dump(info, f)
        logger.info(f"Saved index info to: {info_path}")
    
    def process_csv(self, 
                   csv_path: str, 
                   text_column: str, 
                   output_dir: str = "output",
                   base_name: str = "embeddings",
                   max_rows: Optional[int] = None,
                   batch_size: int = 8,
                   index_type: str = "flat") -> Tuple[faiss.Index, pd.DataFrame]:
        """
        Complete pipeline: load CSV, generate embeddings, create and save FAISS index.
        
        Args:
            csv_path: Path to the CSV file
            text_column: Name of the column containing text to embed
            output_dir: Directory to save the index
            base_name: Base name for the output files
            max_rows: Maximum number of rows to process
            batch_size: Batch size for embedding generation
            index_type: Type of FAISS index to create
            
        Returns:
            Tuple of (FAISS index, DataFrame with original data)
        """
        # Load CSV
        df = self.load_csv(csv_path, text_column, max_rows)
        
        # Get texts to embed
        texts = df[text_column].tolist()
        
        # Generate embeddings
        logger.info("Starting embedding generation...")
        embeddings = self.get_embeddings(texts, batch_size)
        
        # Create FAISS index
        logger.info("Creating FAISS index...")
        index = self.create_faiss_index(embeddings, index_type)
        
        # Save index
        self.save_index(index, output_dir, base_name)
        
        # Save DataFrame with embeddings for reference
        df_path = Path(output_dir) / f"{base_name}_data.csv"
        df.to_csv(df_path, index=False)
        logger.info(f"Saved processed data to: {df_path}")
        
        return index, df


def main():
    """Main function to run the simple CSV embedding pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Load CSV, compute local embeddings, and save FAISS index")
    parser.add_argument("csv_path", help="Path to the CSV file")
    parser.add_argument("text_column", help="Name of the column containing text to embed")
    parser.add_argument("--output-dir", default="output", help="Output directory for index files")
    parser.add_argument("--base-name", default="embeddings", help="Base name for output files")
    parser.add_argument("--max-rows", type=int, help="Maximum number of rows to process")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for embedding generation")
    parser.add_argument("--index-type", default="flat", choices=["flat", "ivf", "hnsw"], 
                       help="Type of FAISS index to create")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", 
                       help="Sentence transformer model name")
    
    args = parser.parse_args()
    
    try:
        # Initialize embedder
        embedder = SimpleCSVEmbedder(model_name=args.model)
        
        # Process CSV
        index, df = embedder.process_csv(
            csv_path=args.csv_path,
            text_column=args.text_column,
            output_dir=args.output_dir,
            base_name=args.base_name,
            max_rows=args.max_rows,
            batch_size=args.batch_size,
            index_type=args.index_type
        )
        
        logger.info("Processing completed successfully!")
        logger.info(f"Index contains {index.ntotal} vectors with dimension {index.d}")
        
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        raise


if __name__ == "__main__":
    main() 