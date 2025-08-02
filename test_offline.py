#!/usr/bin/env python3
"""
Offline Test Script for CSV Embedding and FAISS Indexing

This script tests the system functionality without requiring OpenAI API calls.
It creates dummy embeddings to verify the FAISS indexing and search works.
"""

import os
import sys
import numpy as np
import pandas as pd
import faiss
import pickle
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_dummy_embeddings(texts, dimension=1536):
    """Create dummy embeddings for testing without API calls."""
    embeddings = []
    for i, text in enumerate(texts):
        # Create deterministic dummy embedding based on text length and content
        np.random.seed(hash(text) % 2**32)
        embedding = np.random.normal(0, 1, dimension).astype(np.float32)
        # Normalize to unit vector
        embedding = embedding / np.linalg.norm(embedding)
        embeddings.append(embedding)
    return np.array(embeddings)


def test_csv_loading():
    """Test CSV loading functionality."""
    print("📊 Testing CSV loading...")
    
    try:
        # Load CSV
        df = pd.read_csv("example.csv")
        print(f"✅ Successfully loaded {len(df)} rows from CSV")
        print(f"   Columns: {list(df.columns)}")
        
        # Validate text column
        if "content" in df.columns:
            print("✅ 'content' column found")
            texts = df["content"].tolist()
            print(f"   Sample text: {texts[0][:50]}...")
        else:
            print("❌ 'content' column not found")
            return False
            
        return df, texts
        
    except Exception as e:
        print(f"❌ CSV loading failed: {e}")
        return False


def test_embedding_generation():
    """Test dummy embedding generation."""
    print("\n🧠 Testing embedding generation...")
    
    try:
        # Create sample texts
        sample_texts = [
            "Python is a programming language",
            "Machine learning is a subset of AI",
            "Data analysis with pandas",
            "Web development with Flask",
            "Natural language processing"
        ]
        
        # Generate dummy embeddings
        embeddings = create_dummy_embeddings(sample_texts)
        print(f"✅ Generated {len(embeddings)} embeddings with shape {embeddings.shape}")
        
        return embeddings, sample_texts
        
    except Exception as e:
        print(f"❌ Embedding generation failed: {e}")
        return False


def test_faiss_indexing(embeddings):
    """Test FAISS index creation and search."""
    print("\n🔍 Testing FAISS indexing...")
    
    try:
        # Create FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        index.add(embeddings)
        
        print(f"✅ Created FAISS index with {index.ntotal} vectors")
        
        # Test search
        query_embedding = create_dummy_embeddings(["Python programming"], dimension)[0].reshape(1, -1)
        scores, indices = index.search(query_embedding, k=3)
        
        print(f"✅ Search test successful")
        print(f"   Top scores: {scores[0]}")
        print(f"   Top indices: {indices[0]}")
        
        return index
        
    except Exception as e:
        print(f"❌ FAISS indexing failed: {e}")
        return False


def test_save_load_index(index, output_dir="test_output"):
    """Test saving and loading FAISS index."""
    print("\n💾 Testing index save/load...")
    
    try:
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        
        # Save index
        index_path = Path(output_dir) / "test_index.faiss"
        faiss.write_index(index, str(index_path))
        print(f"✅ Saved index to {index_path}")
        
        # Save metadata
        info = {
            "ntotal": index.ntotal,
            "dimension": index.d,
            "index_type": type(index).__name__
        }
        
        info_path = Path(output_dir) / "test_index_info.pkl"
        with open(info_path, 'wb') as f:
            pickle.dump(info, f)
        print(f"✅ Saved metadata to {info_path}")
        
        # Load index
        loaded_index = faiss.read_index(str(index_path))
        print(f"✅ Loaded index with {loaded_index.ntotal} vectors")
        
        # Test search on loaded index
        query_embedding = create_dummy_embeddings(["test query"], loaded_index.d)[0].reshape(1, -1)
        scores, indices = loaded_index.search(query_embedding, k=2)
        print(f"✅ Search on loaded index successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Save/load test failed: {e}")
        return False


def test_complete_pipeline():
    """Test the complete pipeline with dummy data."""
    print("\n🚀 Testing complete pipeline...")
    
    try:
        # Load CSV
        result = test_csv_loading()
        if not result:
            return False
        df, texts = result
        
        # Generate embeddings
        embeddings = create_dummy_embeddings(texts[:5])  # Use first 5 texts
        
        # Create FAISS index
        index = test_faiss_indexing(embeddings)
        if not index:
            return False
        
        # Save and load index
        if not test_save_load_index(index):
            return False
        
        print("\n✅ Complete pipeline test successful!")
        return True
        
    except Exception as e:
        print(f"❌ Complete pipeline test failed: {e}")
        return False


def main():
    """Main function for offline testing."""
    print("🧪 Offline Testing - CSV Embedding and FAISS Indexing")
    print("=" * 60)
    
    # Check for required files
    if not Path("example.csv").exists():
        print("❌ example.csv not found")
        return
    
    # Run tests
    tests = [
        ("CSV Loading", test_csv_loading),
        ("Embedding Generation", test_embedding_generation),
        ("FAISS Indexing", lambda: test_faiss_indexing(create_dummy_embeddings(["test1", "test2", "test3"]))),
        ("Complete Pipeline", test_complete_pipeline)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 Test Summary:")
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"   {test_name}: {status}")
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    print(f"\n🎯 Results: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n🎉 All tests passed! The system is working correctly.")
        print("\nNote: This was an offline test with dummy embeddings.")
        print("To use real embeddings, you'll need a valid OpenAI API key with sufficient quota.")
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")


if __name__ == "__main__":
    main() 