class Vocabulary:
    def __init__(self, token_to_int=None, unk_token='<UNK>'):
        if token_to_int is None:
            token_to_int = {}
        self._token_to_int = token_to_int
        self._int_to_token = {token_to_int[i]:i for i in token_to_int.keys()}
        self._unk_token = unk_token
        self.unk_int = self.add_token(unk_token)
        
    def add_token(self, token):
        '''updates mapping dictionaries'''
        if token in self._token_to_int:
            index = self._token_to_int[token]
        else:
            index = len(self._token_to_int) # we start indexing tokens from 0
            self._int_to_token[index] = token
            self._token_to_int[token] = index
        return index
    
    def get_serializable_dict(self):
        '''returns a dictionary that can be used in `from_serializable_dict`'''
    
        return {'token_to_int': self._token_to_int,
                'unk_token': self._unk_token}
    
    @classmethod
    def from_serializable_dict(cls, contents):
        '''instantiate a vocabulary from a serialized dictionary'''
        return cls(**contents)
    
    def get_idx_of_token(self, token):
        return self._token_to_int.get(token, unk_int)
    
    def get_token_on_idx(self, idx):
        if idx not in self._int_to_token:
            raise KeyError(f'The index {idx} is not a part of the vocabulary')
        return self._int_to_token.get(idx)
    
    def __str__(self):
        return f'<Vocabulary of size {len(self)}>'
    
    def __len__(self):
        return len(self._token_to_int)
