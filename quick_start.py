#!/usr/bin/env python3
"""
Quick Start Script for CSV Embedding and FAISS Indexing

This script provides an interactive way to get started with the CSV embedding system.
It guides users through the setup process and demonstrates basic functionality.
"""

import os
import sys
from pathlib import Path

def check_setup():
    """Check if the environment is properly set up."""
    print("🔍 Checking setup...")
    
    # Load environment variables first
    from dotenv import load_dotenv
    load_dotenv()
    
    # Check for required files
    required_files = ["embed_csv.py", "search_index.py", "requirements.txt"]
    missing_files = []
    
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {', '.join(missing_files)}")
        return False
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OpenAI API key not found in environment variables.")
        print("   Please set your API key:")
        print("   1. Copy env_example.txt to .env")
        print("   2. Edit .env and add your OpenAI API key")
        print("   3. Get your API key from: https://platform.openai.com/api-keys")
        return False
    
    print("✅ Setup looks good!")
    return True


def install_dependencies():
    """Guide user through dependency installation."""
    print("\n📦 Installing dependencies...")
    
    try:
        import pandas
        import openai
        import faiss
        import numpy
        print("✅ All dependencies are already installed!")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("\nTo install dependencies, run:")
        print("   pip install -r requirements.txt")
        
        response = input("\nWould you like to install dependencies now? (y/n): ").lower()
        if response in ['y', 'yes']:
            os.system("pip install -r requirements.txt")
            return True
        else:
            print("Please install dependencies manually and run this script again.")
            return False


def run_example():
    """Run the example with the provided CSV file."""
    print("\n🚀 Running example with sample data...")
    
    if not Path("example.csv").exists():
        print("❌ example.csv not found. Creating a sample file...")
        create_sample_csv()
    
    try:
        # Import after ensuring dependencies are installed
        from embed_csv import CSVEmbedder
        from search_index import FAISSSearcher
        
        print("📊 Creating embeddings and FAISS index...")
        embedder = CSVEmbedder()
        
        index, df = embedder.process_csv(
            csv_path="example.csv",
            text_column="content",
            output_dir="quick_start_output",
            base_name="quick_start_embeddings",
            max_rows=2,  # Limit for free tier demo
            batch_size=1,
            index_type="flat"
        )
        
        print(f"✅ Created index with {index.ntotal} vectors")
        
        print("\n🔍 Testing search functionality...")
        searcher = FAISSSearcher()
        searcher.load_index("quick_start_output", "quick_start_embeddings")
        
        # Demo search
        query = "Python programming"
        print(f"Searching for: '{query}'")
        results = searcher.search(query, k=3)
        
        for i, (idx, score, data) in enumerate(results, 1):
            print(f"\n{i}. Score: {score:.4f}")
            if data:
                print(f"   Title: {data.get('title', 'N/A')}")
                print(f"   Category: {data.get('category', 'N/A')}")
        
        print("\n✅ Example completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Example failed: {e}")
        return False


def create_sample_csv():
    """Create a sample CSV file if it doesn't exist."""
    sample_data = """id,title,content,category
1,Introduction to Python,Python is a high-level programming language known for its simplicity and readability. It's widely used in data science, web development, and automation.,programming
2,Machine Learning Basics,Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions without being explicitly programmed.,ai
3,Data Analysis with Pandas,Pandas is a powerful Python library for data manipulation and analysis. It provides data structures for efficiently storing large datasets.,data_science
4,Web Development with Flask,Flask is a lightweight web framework for Python that makes it easy to build web applications with minimal boilerplate code.,web_development
5,Natural Language Processing,NLP is a field of AI that focuses on the interaction between computers and human language, enabling machines to understand and generate text.,ai"""
    
    with open("example.csv", "w") as f:
        f.write(sample_data)
    
    print("✅ Created example.csv")


def interactive_demo():
    """Run an interactive demo."""
    print("\n🎯 Interactive Demo")
    print("=" * 30)
    
    try:
        from search_index import FAISSSearcher
        
        searcher = FAISSSearcher()
        searcher.load_index("quick_start_output", "quick_start_embeddings")
        
        print("Enter search queries (type 'quit' to exit):")
        
        while True:
            query = input("\n🔍 Search query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if not query:
                continue
            
            try:
                results = searcher.search(query, k=3)
                print(f"\nResults for '{query}':")
                
                for i, (idx, score, data) in enumerate(results, 1):
                    print(f"\n{i}. Score: {score:.4f}")
                    if data:
                        print(f"   Title: {data.get('title', 'N/A')}")
                        print(f"   Category: {data.get('category', 'N/A')}")
                        print(f"   Content: {data.get('content', 'N/A')[:80]}...")
                        
            except Exception as e:
                print(f"Error during search: {e}")
        
    except Exception as e:
        print(f"❌ Interactive demo failed: {e}")


def main():
    """Main function for the quick start script."""
    print("🚀 CSV Embedding and FAISS Indexing - Quick Start")
    print("=" * 50)
    
    # Check setup
    if not check_setup():
        print("\n❌ Setup incomplete. Please fix the issues above and run again.")
        return
    
    # Install dependencies
    if not install_dependencies():
        return
    
    # Run example
    if not run_example():
        return
    
    # Offer interactive demo
    print("\n" + "=" * 50)
    print("🎉 Quick start completed successfully!")
    print("\nNext steps:")
    print("1. Try the interactive demo")
    print("2. Process your own CSV file")
    print("3. Explore the documentation")
    
    response = input("\nWould you like to try the interactive demo? (y/n): ").lower()
    if response in ['y', 'yes']:
        interactive_demo()
    
    print("\n📚 For more information, see the README.md file.")
    print("🔧 For advanced usage, run: python embed_csv.py --help")


if __name__ == "__main__":
    main() 