# Text Dissimilarity Analyzer

![Project Banner](https://via.placeholder.com/1200x300.png?text=Text+Dissimilarity+Analyzer)

A powerful web application built with Streamlit to analyze text dissimilarity using **Word Mover's Distance (WMD)**, leveraging Google's word2vec embeddings and Gurobi optimization. This project supports single pair comparison, batch file comparison (dissimilarity matrix), and plagiarism detection, with enhanced visualizations including word clouds, frequency bar plots, word flow graphs, and dissimilarity heatmaps.

## Features

- **Single Pair Comparison**: Compare two text documents to compute their WMD score, with visualizations like word clouds, word frequency bar plots, and word flow graphs.
- **Batch File Comparison**: Upload multiple `.txt` files to generate a pairwise dissimilarity matrix, visualized as a heatmap and bar plot of top dissimilarities.
- **Plagiarism Detection**: Compare a sample sentence against a book to find the closest matching sentence, useful for detecting rephrased content.
- **Advanced Preprocessing**: Uses lemmatization (via NLTK's WordNetLemmatizer) and removes stop words, punctuation, and proper nouns.
- **Out-of-Vocabulary Handling**: Approximates embeddings for words not in word2vec using similar words' embeddings.
- **Enhanced Visualizations**:
  - Word clouds for processed documents.
  - Word frequency bar plots for single pair comparison.
  - Directed word flow graphs with edge weights proportional to flow values.
  - Heatmap and bar plot for batch comparison dissimilarity scores.
- **Interactive Streamlit Interface**: Modern, user-friendly UI with a blue-orange-gray color scheme, large fonts, and organized layout.

## Screenshots

| **Single Pair Comparison** | **Batch File Comparison** | **Plagiarism Detection** |
|----------------------------|---------------------------|--------------------------|
| ![Single Pair](https://via.placeholder.com/600x400.png?text=Single+Pair+Comparison) | ![Batch Comparison](https://via.placeholder.com/600x400.png?text=Batch+Comparison+Heatmap) | ![Plagiarism Detection](https://via.placeholder.com/600x400.png?text=Plagiarism+Detection) |

*Note: Replace placeholder images with actual screenshots after running the app.*

## Prerequisites

- Python 3.8+
- A Gurobi license (free academic license available at [gurobi.com](https://www.gurobi.com/))
- Internet connection for initial word2vec model download

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/text-dissimilarity-analyzer.git
   cd text-dissimilarity-analyzer
