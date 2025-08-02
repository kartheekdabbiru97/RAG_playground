#!/usr/bin/env python3
"""
Free Tier Demo Script for CSV Embedding and FAISS Indexing

This script is specifically designed for OpenAI free tier users with limited API calls.
It processes only 2 rows and uses minimal batch sizes to stay within free tier limits.
"""

import os
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from embed_csv import CSVEmbedder
from search_index import FAISSSearcher


def main():
    """Run a minimal demo for free tier users."""
    print("🚀 Free Tier Demo - CSV Embedding and FAISS Indexing")
    print("=" * 55)
    print("⚠️  This demo is optimized for OpenAI free tier limits")
    print("   - Processes only 2 rows")
    print("   - Uses batch size of 1")
    print("   - Minimal API calls")
    print("=" * 55)
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OpenAI API key not found.")
        print("   Please set your API key in the .env file")
        return
    
    try:
        # Step 1: Create embeddings and index (minimal)
        print("\n📊 Step 1: Creating embeddings (2 rows only)...")
        embedder = CSVEmbedder()
        
        index, df = embedder.process_csv(
            csv_path="example.csv",
            text_column="content",
            output_dir="free_tier_output",
            base_name="free_tier_embeddings",
            max_rows=2,  # Only 2 rows for free tier
            batch_size=1,  # Process one at a time
            index_type="flat"
        )
        
        print(f"✅ Created index with {index.ntotal} vectors")
        
        # Step 2: Test search functionality
        print("\n🔍 Step 2: Testing search functionality...")
        searcher = FAISSSearcher()
        searcher.load_index("free_tier_output", "free_tier_embeddings")
        
        # Test with a simple query
        query = "Python programming"
        print(f"Searching for: '{query}'")
        results = searcher.search(query, k=2)
        
        for i, (idx, score, data) in enumerate(results, 1):
            print(f"\n{i}. Score: {score:.4f}")
            if data:
                print(f"   Title: {data.get('title', 'N/A')}")
                print(f"   Category: {data.get('category', 'N/A')}")
                print(f"   Content: {data.get('content', 'N/A')[:60]}...")
        
        print("\n✅ Free tier demo completed successfully!")
        print("\n📊 Summary:")
        print(f"   - Processed {len(df)} rows")
        print(f"   - Created FAISS index with {index.ntotal} vectors")
        print(f"   - Used minimal API calls for free tier")
        
        print("\n🎯 Next steps:")
        print("1. Try your own small CSV file (2-3 rows)")
        print("2. Use '--max-rows 2' to limit processing")
        print("3. Use '--batch-size 1' for minimal API calls")
        print("4. Consider upgrading for larger datasets")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("\n💡 Tips for free tier:")
        print("   - Keep datasets small (2-5 rows)")
        print("   - Use batch_size=1")
        print("   - Check your API usage at: https://platform.openai.com/account/usage")


if __name__ == "__main__":
    main() 