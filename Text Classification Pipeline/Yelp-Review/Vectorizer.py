# counts the frequency of all tokens present in the dataset and encapsulates the Vocabulary class to create a vocabulary (from the dataframe)
class ReviewVectorizer:
    def __init__(vocab):
        self.vocab = vocab
    
    # convert a review (string) into a one hot vector representation
    def vectorize(self, review):
        one_hot = np.zeroes(len(self.vocab))
        for token in review.split():
            if token not in string.punctuation:
                one_hot[self.vocab.lookup_int_of_token(token)] = 1
        return one_hot
    
    @classmethod
    def from_dataframe(cls, review_df, cutoff=25):
        vocab = Vocabulary()
        word_freq = Counter()
        for review in review_df['review']:
            for token in review.split():
                if token not in string.punctuation:
                    word_freq[token] += 1
                    
        for token, count in word_freq.items():
            if count > cutoff:
                vocab.add_token(token)
        return cls(vocab)
    
    @classmethod
    def from_serializable(self, contents):
        vocab = Vocabulary.from_serializable_dict(contents['vocab'])
        return cls(vocab=vocab)
    
    def to_serializable(self):
        return {'vocab': self.vocab.get_serializable_dict()}
