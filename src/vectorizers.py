import math
from collections import Counter

class TextVectorizer:
    def __init__(self, method="tf"):
        """
        Initializes the vectorizer.
        :param method: Parameterization algorithm to use. 
                       Valid options: 'binary', 'tf', 'tfidf'
        """
        if method not in ["binary", "tf", "tfidf"]:
            raise ValueError("Method must be 'binary', 'tf', or 'tfidf'")
        
        self.method = method
        self.vocabulary = {}
        self.vocab_size = 0
        self.document_frequencies = {} # How many docs contain the word 't'
        self.total_docs = 0
        self.idf = {}

    def fit(self, raw_documents):
        """
        Builds the vocabulary from the training documents.
        """
        self.total_docs = len(raw_documents)
        all_words = set()
        
        for doc in raw_documents:
            # Convert to lowercase and split by spaces (basic tokenization)
            words = doc.lower().split()
            unique_words_in_doc = set(words)
            
            # For IDF calculation: count in how many different documents each word appears
            for word in unique_words_in_doc:
                self.document_frequencies[word] = self.document_frequencies.get(word, 0) + 1
                
            # Add words to the global set
            all_words.update(unique_words_in_doc)
            
        # Assign a numeric index to each unique word (alphabetically sorted)
        self.vocabulary = {word: idx for idx, word in enumerate(sorted(all_words))}
        self.vocab_size = len(self.vocabulary)
        
        # If the method is TF-IDF, precalculate IDF values to save time later
        if self.method == "tfidf":
            for word, doc_freq in self.document_frequencies.items():
                # Standard IDF formula: log( N / df )
                self.idf[word] = math.log(self.total_docs / float(doc_freq))

    def transform(self, raw_documents):
        """
        Transforms a list of texts into a list of numeric vectors using the learned vocabulary.
        """
        vectors = []
        for doc in raw_documents:
            words = doc.lower().split()
            word_counts = Counter(words)
            
            # Initialize a vector of zeros the size of our vocabulary
            vector = [0.0] * self.vocab_size
            
            for word, count in word_counts.items():
                # Only process the word if we saw it during training (fit)
                if word in self.vocabulary:
                    idx = self.vocabulary[word]
                    
                    # 1. Binary: 1 if it appears, 0 if not
                    if self.method == "binary":
                        vector[idx] = 1.0
                        
                    # 2. Term Frequency (TF): Absolute frequency
                    elif self.method == "tf":
                        vector[idx] = float(count)
                        
                    # 3. TF-IDF: Frequency * Inverse Document Frequency
                    elif self.method == "tfidf":
                        tf = float(count)
                        vector[idx] = tf * self.idf.get(word, 0.0)
            
            vectors.append(vector)
            
        return vectors

    def fit_transform(self, raw_documents):
        """
        Learns the vocabulary and transforms the texts in a single step.
        """
        self.fit(raw_documents)
        return self.transform(raw_documents)
    