# Automated mining for figures and tables in papers and reports

## Project Objective:

To develop an advanced system to automate extraction of figures and tables from academic documents and generate semantic classification and analysis of figures and tables across multiple domains.

## Environment setup

Make sure your python version is 3.6 or later.
Install dependencies with pip:

```
pip install -r requirements.txt
```

## PDFFigures 2

PDFFigures 2 is employed for initial data gathering from academic documents, feeding into further semantic analysis and classification in our system.

### Usage

For batch processing of PDFs and saving the extracted figures and statistics, use the following command:

```
sudo sbt "runMain org.allenai.pdffigures2.FigureExtractorBatchCli /path/to/pdf_directory/ -s stat_file.json -m /figure/image/output/prefix -d /figure/data/output/prefix"
```

## Classification model

## LLaVA

Make sure to mount the images from /src/figure_classification/test/images to your drive 
then run the ipynb file with T4 GPU on google colab

You can find the ipynb file here: /src/LLaVA.ipynb

## Word Embedding Enhancement with Word2Vec

This updated script enhances datasets with Word2Vec word embeddings and NLTK-based preprocessed text descriptions. For datasets containing figure descriptions, this process enriches them with numerical vectors suitable for further analysis.

### Usage

Run the script with your dataset to generate word embeddings:

```
python src/data_preprocessing_word_embedding.py -dataset path/to/your/dataset.csv -output_dataset /path/to/output.csv
```

This command preprocesses the text descriptions in your dataset and appends Word2Vec embeddings, storing the output in the specified file.

## Similarity Matrix Calculation

This script calculates a combined similarity matrix from datasets containing preprocessed descriptions and their corresponding Word2Vec embeddings. It is a crucial step in analyzing the textual and semantic relationships between different figures in the dataset.

### Usage

Execute the script with the command below to compute the similarity matrix:

```
python src/similarity_scores_calculation.py -dataset path/to/your/dataset.csv -output_dataset /path/to/output/matrix.csv
```

### Description

- The script uses both TF-IDF and Word2Vec vector representations to compute similarity scores.
- The output is a CSV file with the combined cosine similarity matrix, offering insights into the similarity between different text entries based on their embeddings and TF-IDF scores.

## References

This project makes use of pdffigures2, an open-source tool developed by the Allen Institute for AI for extracting figures, tables, and captions from scholarly documents. For more information on pdffigures2, visit their GitHub repository:

[pdffigures2 GitHub Repository](https://github.com/allenai/pdffigures2)
