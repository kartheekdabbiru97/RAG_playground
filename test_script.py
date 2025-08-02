#!/usr/bin/env python3
"""
Test script to demonstrate the CSV embedding and FAISS indexing functionality.
This script uses the example.csv file to show how the system works.
"""

import os
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from embed_csv import CSVEmbedder
from search_index import FAISSSearcher


def test_embedding_pipeline():
    """Test the complete embedding and search pipeline."""
    print("🚀 Testing CSV Embedding and FAISS Indexing Pipeline")
    print("=" * 60)
    
    # Check if example.csv exists
    if not Path("example.csv").exists():
        print("❌ example.csv not found. Please ensure the example file is present.")
        return False
    
    try:
        # Step 1: Create embeddings and index
        print("\n📊 Step 1: Creating embeddings and FAISS index...")
        embedder = CSVEmbedder()
        
        index, df = embedder.process_csv(
            csv_path="example.csv",
            text_column="content",
            output_dir="test_output",
            base_name="test_embeddings",
            max_rows=5,  # Limit for testing
            batch_size=2,
            index_type="flat"
        )
        
        print(f"✅ Created index with {index.ntotal} vectors")
        
        # Step 2: Test search functionality
        print("\n🔍 Step 2: Testing search functionality...")
        searcher = FAISSSearcher()
        searcher.load_index("test_output", "test_embeddings")
        
        # Test queries
        test_queries = [
            "Python programming language",
            "machine learning algorithms",
            "data analysis tools",
            "web development frameworks"
        ]
        
        for query in test_queries:
            print(f"\n🔎 Searching for: '{query}'")
            results = searcher.search(query, k=3)
            
            for i, (idx, score, data) in enumerate(results, 1):
                print(f"  {i}. Score: {score:.4f}")
                if data:
                    print(f"     Title: {data.get('title', 'N/A')}")
                    print(f"     Category: {data.get('category', 'N/A')}")
                    print(f"     Content: {data.get('content', 'N/A')[:100]}...")
        
        print("\n✅ All tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False


def test_programmatic_usage():
    """Test programmatic usage of the classes."""
    print("\n🔧 Testing Programmatic Usage")
    print("=" * 40)
    
    try:
        # Test CSVEmbedder class
        print("Testing CSVEmbedder class...")
        embedder = CSVEmbedder()
        
        # Test loading CSV
        df = embedder.load_csv("example.csv", "content", max_rows=3)
        print(f"✅ Loaded {len(df)} rows from CSV")
        
        # Test getting embeddings
        texts = df["content"].tolist()
        embeddings = embedder.get_embeddings(texts, batch_size=2)
        print(f"✅ Generated {len(embeddings)} embeddings with shape {embeddings.shape}")
        
        # Test creating FAISS index
        index = embedder.create_faiss_index(embeddings, "flat")
        print(f"✅ Created FAISS index with {index.ntotal} vectors")
        
        # Test FAISSSearcher class
        print("\nTesting FAISSSearcher class...")
        searcher = FAISSSearcher()
        
        # Test getting single embedding
        query_embedding = searcher.get_embedding("test query")
        print(f"✅ Generated query embedding with shape {query_embedding.shape}")
        
        # Test search by embedding
        results = searcher.search_by_embedding(query_embedding, k=2)
        print(f"✅ Search by embedding returned {len(results)} results")
        
        print("✅ Programmatic usage tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Programmatic usage test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🧪 Running CSV Embedding and FAISS Indexing Tests")
    print("=" * 60)
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not set. Tests will fail.")
        print("   Please set your OpenAI API key in the .env file or environment.")
        print("   You can get an API key from: https://platform.openai.com/api-keys")
        return
    
    # Run tests
    test1_success = test_embedding_pipeline()
    test2_success = test_programmatic_usage()
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 Test Summary:")
    print(f"   Pipeline Test: {'✅ PASSED' if test1_success else '❌ FAILED'}")
    print(f"   Programmatic Test: {'✅ PASSED' if test2_success else '❌ FAILED'}")
    
    if test1_success and test2_success:
        print("\n🎉 All tests passed! The system is working correctly.")
        print("\nNext steps:")
        print("1. Try processing your own CSV file:")
        print("   python embed_csv.py your_file.csv your_text_column")
        print("2. Search the created index:")
        print("   python search_index.py test_output")
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")


if __name__ == "__main__":
    main() 