# Hate Speech Detection using NLP and Machine Learning

## English

### Project Description
This project uses machine learning and natural language processing (NLP) to classify hate speech in social media posts. It involves preprocessing text data, feature extraction using TF-IDF, dimensionality reduction, oversampling with SMOTE, and training a neural network (MLPClassifier). The dataset contains posts labeled as "normal", "offensive", or "hate speech".

### Key Libraries
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical operations
- **NLTK**: Natural language processing (tokenization, stopwords, lemmatization)
- **SpaCy**: Named Entity Recognition (NER)
- **Scikit-learn**: Machine learning tools for classification, feature extraction (TF-IDF), and dimensionality reduction (SVD)
- **Imbalanced-learn**: SMOTE for oversampling imbalanced datasets
- **Seaborn/Matplotlib**: Data visualization
- **SMOTE**: Handling class imbalance with synthetic oversampling.

### Key Steps
1. **Data Preprocessing**: Remove special characters, tokenize, remove stopwords, and apply lemmatization and stemming. Named Entity Recognition (NER) ensures that named entities are preserved.
2. **Text Vectorization**: TF-IDF vectorizer converts text data into numerical features.
3. **Dimensionality Reduction**: SVD reduces the feature space to 300 dimensions.
4. **Handling Imbalanced Data**: SMOTE oversamples minority classes to handle class imbalance.
5. **Model Training**: The MLPClassifier is trained using stratified cross-validation and optimized to classify text into "normal", "offensive", and "hate speech".
6. **Evaluation**: The model performance is evaluated using classification reports and confusion matrices.

### Installation
1. Install the required Python libraries:
   ```bash
   pip install pandas numpy nltk spacy scikit-learn imbalanced-learn seaborn matplotlib

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

python -m spacy download en_core_web_sm
