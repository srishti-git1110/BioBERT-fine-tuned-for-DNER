import torch
from torch.utils.data import Dataset, DataLoader

class YelpReview(Dataset):
    def __init__(self, review_df, vectorizer):
        self.df = review_df
        self._vectorizer = vectorizer
        
        self.train_data = self.df[self.df['split']=='train']
        self.train_size = len(self.train_data)
        
        self.val_data = self.df[self.df['split']=='validation']
        self.val_size = len(self.val_data)
        
        self.test_data = self.df[self.df['split']=='test']
        self.test_size = len(self.test_data)
        
        self._lookup_dict = {'train': (self.train_df, self.train_size),
                             'val': (self.val_df, self.validation_size),
                             'test': (self.test_df, self.test_size)}
        self.set_split('train')
        
    @classmethod
    def load_and_vectorize_data(cls, csv_file):
        review_df = pd.read_csv(csv_file)
        return cls(review_df, ReviewVectorizer.from_dataframe(review_df))
    
    def get_vectorizer(self):
        return self._vectorizer
    
    def set_split(self, split="train"):
        self._target_split = split
        self._target_df, self._target_size = self._lookup_dict[split]
        
    def __len__():
        return self._target_size
    
    def __getitem__(self, index):
        label = review_df.iloc[index, 'class']  # need to change
        review = self._vectorizer.vectorize(review_df.iloc[index, 'review'])
        return {'x': review,
                'y': label} 
    
    def num_of_batches(self, batch_size):
        return len(self) // batch_size
