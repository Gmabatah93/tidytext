# Tidytext Format

<img src="Images/tidytext.PNG" width="700">

**Token:** a meaningful unit of text, most often a word, that we are interested in using for further analysis, and tokenization is the process of splitting text into tokens.\
**Corpus:** raw strings annotated with additional metadata and details.\
**Document-Term Matrix:** a sparse matrix describing a collection (i.e., a corpus) of documents with one row for each document and one column for each term. The value in the matrix is typically word count or tf-idf.

# Sentiment Analysis

<img src="Images/sentiment.PNG" width="700">

**_DATASETS_**
- **AFINN** _assigns words with a score that runs between -5 and 5, with negative scores indicating negative sentiment and positive scores indicating positive sentiment._
- **bing:** _categorizes words in a binary fashion into positive and negative categories._
- **nrc:** _categorizes words in a binary fashion [ yes/no ] into categories of positive, negative, anger, anticipation, disgust, fear, joy, sadness, surprise, and trust._

# Term-Frequency Inverse Document Frequency
> **Term Frequency:** how frequently a word occurs in a document.\
**Inverse Document Frequency:**  decreases the weight for commonly used words and increases the weight for words that are not used very much in a collection of documents.\
**TF-IDF:** the frequency of a term adjusted for how rarely it is used. _(find the important words for the content of each document by decreasing the weight for commonly used words and increasing the weight for words that are not used very much in a collection or corpus of documents)_

<img src="Images/idf.PNG" width="500">

# n-grams & Correlations

# Converting to and from non-tidy formats
<img src="Images/nontidy.PNG" width="700">

## Document-Term Matrix
> - Each row represents one document
> - Each column represents one term
> - Each value (typically) contains the number of appearances of that term in that document
