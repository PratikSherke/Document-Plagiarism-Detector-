```markdown
# Text Dissimilarity Analyzer

![Text Dissimilarity Analyzer Banner](https://via.placeholder.com/800x200.png?text=Text+Dissimilarity+Analyzer)  
*Compare text documents with style and precision using Word Mover's Distance (WMD)*

**Text Dissimilarity Analyzer** is a powerful Streamlit-based web application designed to measure semantic dissimilarity between text documents using **Word Mover's Distance (WMD)**. Leveraging Python, Gurobi, and Google's pre-trained word2vec model, this tool excels in applications like plagiarism detection, text comparison, and semantic analysis. With an intuitive interface and stunning visualizations—including word clouds, word flow graphs, dissimilarity matrices, and bar plots—it makes text analysis both insightful and engaging.

## Features

- **Single Pair Comparison**: Compute WMD scores for two documents, with visualizations like word clouds, word frequency bar plots, and directed word flow graphs.
- **Batch File Comparison**: Upload multiple `.txt` files to generate a pairwise dissimilarity score matrix, visualized as a heatmap and bar plot of top pairs, with downloadable CSV results.
- **Plagiarism Detection**: Identify the closest matching sentence in a book for a given sample text, perfect for detecting rephrased or plagiarized content.
- **Advanced Preprocessing**: Uses lemmatization (via NLTK's WordNetLemmatizer) and handles out-of-vocabulary words by approximating embeddings.
- **Enhanced Visualizations**:
  - **Word Flow Graph**: Interactive graph showing word mappings with flow weights, styled with clear layouts and vibrant colors (using `networkx`).
  - **Dissimilarity Matrix**: Heatmap of pairwise WMD scores for batch comparisons, using a yellow-orange-red color scheme.
  - **Word Clouds and Frequency Plots**: Visualize word distributions for intuitive insights.
- **Modern Interface**: Sleek Streamlit UI with a blue-orange color scheme, large readable fonts, and organized layout for a professional user experience.

## Demo

![Single Pair Comparison Demo](https://via.placeholder.com/600x300.png?text=Single+Pair+Comparison+Demo)  
*Compare two documents with word clouds, frequency plots, and word flow visualization.*

![Batch Comparison Heatmap](https://via.placeholder.com/600x300.png?text=Batch+Comparison+Heatmap)  
*Visualize dissimilarity scores for multiple documents as a heatmap and bar plot.*

## Installation

### Prerequisites
- Python 3.8+
- A valid [Gurobi license](https://www.gurobi.com/free-trial/) (free for academic use)
- Internet connection for initial word2vec model download

### Steps
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/text-dissimilarity-analyzer.git
   cd text-dissimilarity-analyzer
   ```

2. **Install Dependencies**:
   Create a virtual environment (optional but recommended) and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Run the Application**:
   ```bash
   streamlit run text_dissimilarity_app.py
   ```
   Open your browser at `http://localhost:8501` to access the app.

## Usage

### 1. Single Pair Comparison
- Select **Single Pair Comparison** from the sidebar.
- Enter two documents in the text areas (e.g., sentences or paragraphs).
- Click **Compute WMD** to view:
  - Dissimilarity score in a styled box
  - Word clouds and frequency bar plots for each document
  - Word flow graph showing semantic mappings between words
  - Detailed flow table with word pairs and distances

### 2. Batch File Comparison
- Select **Batch File Comparison** from the sidebar.
- Upload multiple `.txt` files, each containing a document or sentence (e.g., use `sample_data.txt`).
- Click **Compute Dissimilarity Matrix** to see:
  - A heatmap of pairwise WMD scores
  - A bar plot of the top 10 lowest dissimilarity scores
  - A downloadable CSV of the dissimilarity matrix
  - Select a pair from a dropdown to view its word flow graph

### 3. Plagiarism Detection
- Select **Plagiarism Detection** from the sidebar.
- Enter a sample sentence and upload a book text file (e.g., from [Project Gutenberg](https://www.gutenberg.org/)).
- Click **Find Closest Match** to identify the most similar sentence in the book, with a table of top matching sentences.

### Sample Files
- `sample_data.txt`: A sample text file with sentences for testing batch comparison or plagiarism detection.

## Dependencies
Listed in `requirements.txt`:
```
numpy
pandas
matplotlib
seaborn
gensim
scipy
nltk
wordcloud
gurobipy
streamlit
networkx
```

## Project Structure
```
text-dissimilarity-analyzer/
├── text_dissimilarity_app.py  # Main Streamlit application
├── requirements.txt          # Dependencies
├── sample_data.txt          # Sample text file for testing
└── README.md                # Project documentation
```

## Contributing
We welcome contributions! To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -m 'Add your feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a Pull Request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Built with [Streamlit](https://streamlit.io/), [Gurobi](https://www.gurobi.com/), and [Google's word2vec](https://code.google.com/archive/p/word2vec/).
- Inspired by the [Gurobi text dissimilarity example](https://github.com/Gurobi/modeling-examples).
- Uses [Project Gutenberg](https://www.gutenberg.org/) for sample texts.

## Contact
For questions or feedback, open an issue on GitHub or contact [your-email@example.com].

*Happy analyzing!*
```