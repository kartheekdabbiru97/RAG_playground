# CSV Embedding and FAISS Indexing

This project provides Python scripts to load CSV data, compute sentence embeddings using OpenAI API, and save the index in FAISS format for efficient similarity search.

## Features

- **CSV Loading**: Load and preprocess CSV files with automatic validation
- **OpenAI Embeddings**: Generate high-quality embeddings using OpenAI's embedding models
- **FAISS Indexing**: Create efficient similarity search indexes with multiple index types
- **Batch Processing**: Process large datasets efficiently with configurable batch sizes
- **Search Interface**: Interactive and programmatic search capabilities
- **Flexible Output**: Save indexes and metadata for later use

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd ssmvp_rag
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your OpenAI API key:
```bash
# Copy the example environment file
cp env_example.txt .env

# Edit .env and add your OpenAI API key
# Get your API key from: https://platform.openai.com/api-keys
```

## Usage

### 1. Create Embeddings and FAISS Index

The main script `embed_csv.py` processes CSV files and creates FAISS indexes:

```bash
python embed_csv.py <csv_file> <text_column> [options]
```

#### Basic Usage:
```bash
# Process a CSV file with default settings
python3 embed_csv.py example.csv content

# Process with custom output directory and name
python3 embed_csv.py example.csv content --output-dir my_index --base-name my_embeddings

# Limit processing to first 100 rows (for testing)
python3 embed_csv.py example.csv content --max-rows 100

# Use different FAISS index type
python3 embed_csv.py example.csv content --index-type hnsw

# Free tier usage (minimal API calls)
python3 embed_csv.py example.csv content --max-rows 2 --batch-size 1
```

#### Command Line Options:
- `csv_file`: Path to the CSV file
- `text_column`: Name of the column containing text to embed
- `--output-dir`: Output directory for index files (default: "output")
- `--base-name`: Base name for output files (default: "embeddings")
- `--max-rows`: Maximum number of rows to process (for testing)
- `--batch-size`: Batch size for embedding generation (default: 100)
- `--index-type`: Type of FAISS index ("flat", "ivf", "hnsw") (default: "flat")
- `--model`: OpenAI embedding model (default: "text-embedding-ada-002")

### 2. Search the Index

Use `search_index.py` to search the created FAISS index:

```bash
python search_index.py <index_directory> [options]
```

#### Basic Usage:
```bash
# Interactive search mode
python search_index.py output

# Single query search
python search_index.py output --query "machine learning"

# Search with custom parameters
python search_index.py output --query "data analysis" --k 10
```

#### Command Line Options:
- `index_directory`: Directory containing the FAISS index files
- `--base-name`: Base name of the index files (default: "embeddings")
- `--query`: Query text to search for (if not provided, runs interactively)
- `--k`: Number of results to return (default: 5)
- `--model`: OpenAI embedding model (default: "text-embedding-ada-002")

## FAISS Index Types

The script supports three types of FAISS indexes:

1. **Flat Index** (`--index-type flat`):
   - Exact search, 100% accuracy
   - Good for small to medium datasets (< 1M vectors)
   - Slower for large datasets

2. **IVF Index** (`--index-type ivf`):
   - Approximate search, high accuracy
   - Good for large datasets (> 100K vectors)
   - Requires training on the data

3. **HNSW Index** (`--index-type hnsw`):
   - Approximate search, good accuracy
   - Good balance of speed and accuracy
   - Works well for most use cases

## Example Workflow

1. **Prepare your data**:
   ```bash
   # Your CSV should have a column with text to embed
   # Example: id,title,content,category
   ```

2. **Create embeddings and index**:
   ```bash
   # For free tier users (minimal API calls)
   python3 embed_csv.py example.csv content --max-rows 2 --batch-size 1
   
   # For paid users
   python3 embed_csv.py example.csv content --output-dir my_index
   ```

3. **Search the index**:
   ```bash
   # Interactive mode
   python3 search_index.py my_index
   
   # Or single query
   python3 search_index.py my_index --query "Python programming"
   ```

## Output Files

The script creates several output files:

- `{base_name}.faiss`: The FAISS index file
- `{base_name}_info.pkl`: Index metadata (dimensions, type, etc.)
- `{base_name}_data.csv`: Original data with processed rows

## Programmatic Usage

You can also use the classes programmatically:

```python
from embed_csv import CSVEmbedder
from search_index import FAISSSearcher

# Create embeddings
embedder = CSVEmbedder()
index, df = embedder.process_csv(
    csv_path="data.csv",
    text_column="content",
    output_dir="output",
    index_type="hnsw"
)

# Search the index
searcher = FAISSSearcher()
searcher.load_index("output")
results = searcher.search("your query here", k=5)

for idx, score, data in results:
    print(f"Score: {score:.4f}")
    print(f"Data: {data}")
```

## Free Tier Usage

If you're using OpenAI's free tier, you have limited API calls. Here are tips to stay within limits:

### Free Tier Optimizations:
```bash
# Use minimal settings for free tier
python3 embed_csv.py example.csv content --max-rows 2 --batch-size 1

# Or use the free tier demo script
python3 free_tier_demo.py
```

### Free Tier Limits:
- **API Calls**: Very limited (check your usage at https://platform.openai.com/account/usage)
- **Recommended**: Process 2-5 rows at a time
- **Batch Size**: Use 1 to minimize API calls
- **Testing**: Use the offline test script first: `python3 test_offline.py`

### Free Tier Demo:
```bash
# Run the optimized free tier demo
python3 free_tier_demo.py
```

## Configuration

### Environment Variables

Create a `.env` file with your configuration:

```bash
# Required
OPENAI_API_KEY=your_api_key_here

# Optional
OPENAI_MODEL=text-embedding-ada-002
```

### Batch Size Optimization

- **Small datasets** (< 1K rows): Use batch_size=50-100
- **Medium datasets** (1K-10K rows): Use batch_size=100-200
- **Large datasets** (> 10K rows): Use batch_size=200-500

### Index Type Selection

- **Small datasets** (< 10K vectors): Use `flat`
- **Medium datasets** (10K-1M vectors): Use `hnsw`
- **Large datasets** (> 1M vectors): Use `ivf`

## Error Handling

The scripts include comprehensive error handling:

- **API Rate Limits**: Automatic retry logic for OpenAI API calls
- **Invalid Data**: Skips rows with empty or invalid text
- **File Validation**: Checks for required files and directories
- **Memory Management**: Processes data in batches to avoid memory issues

## Troubleshooting

### Common Issues

1. **OpenAI API Key Error**:
   ```
   ValueError: OpenAI API key is required
   ```
   Solution: Set your API key in the `.env` file or pass it as a parameter.

2. **CSV Column Not Found**:
   ```
   ValueError: Text column 'content' not found in CSV
   ```
   Solution: Check your CSV file and specify the correct column name.

3. **Memory Issues with Large Files**:
   Solution: Reduce batch size or use `--max-rows` to process a subset first.

4. **FAISS Index Type Error**:
   ```
   ValueError: Unsupported index type
   ```
   Solution: Use one of the supported types: "flat", "ivf", "hnsw".

## Performance Tips

1. **Use appropriate index types** for your dataset size
2. **Optimize batch sizes** based on your data volume
3. **Process in chunks** for very large datasets
4. **Use GPU acceleration** if available (requires faiss-gpu)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 