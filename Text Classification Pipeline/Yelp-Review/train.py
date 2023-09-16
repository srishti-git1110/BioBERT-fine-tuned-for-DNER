import numpy as np 
import pandas as pd
import collections
import re
from argparse import Namespace
import torch.optim as optim
from Dataset import YelpReview
from model import Model


train = pd.read_csv('../input/yelp-review-dataset/yelp_review_polarity_csv/train.csv', header=None)
train.rename(columns={0:'class',
                     1:'review'}, inplace=True)
subset = train.iloc[:56000,:]


by_rating = collections.defaultdict(list)
for _, row in subset.iterrows():
    by_rating[row['class']].append(row.to_dict())
    
final_list = []

for class_label, row_dict_list in (sorted(by_rating.items())):
    np.random.shuffle(row_dict_list)
    n_tot = len(row_dict_list)
    
    n_train = int(0.6*n_tot)
    n_val = int(.25*n_tot)
    n_test = int(.15*n_tot)
    
    for row_dict in row_dict_list[:n_train]:
        row_dict['split'] = 'train'
    for row_dict in row_dict_list[n_train:n_val+n_train]:
        row_dict['split'] = 'validation'
    for row_dict in row_dict_list[n_val+n_train:n_test+n_train+n_val]:
        row_dict['split'] = 'test'
      
    final_list.extend(row_dict_list)
review_df = pd.DataFrame(final_list) # final_list is a list of dictionaries
review_df.head()


def preprocess_reviews(text):
    text = text.lower()
    text = re.sub(r'([.;@,?!&])', r" \1 ",text)
    text = re.sub(r"[^a-zA-Z,.;@#?!&$]+", r" ", text)
    return text
review_df['review'] = review_df['review'].apply(preprocess_reviews)
review_df.to_csv('../input/yelp-review-dataset/yelp_review_polarity_csv/train1.csv')




def generate_batches(dataset, batch_size, shuffle=True, drop_last=True, device='cpu'):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    for data_dict in dataloader:
        out_dict = {}
        for name, tensor in data_dict.items():
            out_dict[name] = data_dict[name].to(device)
        yield out_dict
        

args = Namespace(
    cutoff=25,
    model_file='yelp_model.pth',
    review_csv='../input/yelp-review-dataset/yelp_review_polarity_csv/train1.csv',
    vectorizer_file='vectorizer.json',
    batch_size=64,
    learning_rate=.005,
    num_epochs=80,
    early_stopping_criterion=5
    )


def make_train_state(args):
    return {'epoch': 0,
           'train_loss': [],
           'train_acc': [],
           'val_loss': [],
           'val_acc': [],
           'test_loss': -1,
           'test_acc': -1}

train_state = make_train_state(args)
args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = YelpReview.load_and_vectorize_data(csv_file='../input/yelp-review-dataset/yelp_review_polarity_csv/train1.csv')
vectorizer = dataset.get_vectorizer()
classifier = (Model(len(vectorizer.vocab))).to(device=args.device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(classifier.parameters, lr=args.learning_rate)

for epoch in range(args.num_epochs):
    
    # train
    train_state['epoch'] = epoch
    dataset.set_split('train')
    classifier.train()
    batch_generator = generate_batches(dataset, batch_size=args.batch_size, device=args.device) 
    running_mean_loss = 0.0
    running_mean_acc = 0.0
    for batch_idx, batch_dict in enumerate(batch_generator):
        y_pred = classifier(batch_dict['x'])
        loss = criterion(y_pred, batch_dict['y'])
        
        batch_loss = loss.item()
        batch_acc = compute_accuracy(y_pred, batch_dict['y'])
        running_mean_loss += (batch_loss - running_mean_loss) / (batch_idx + 1)
        running_mean_acc += (batch_acc - running_mean_acc) / (batch_idx + 1)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    train_state['train_loss'].append(running_mean_loss)
    train_state['train_acc'].append(running_mean_acc)
    
    # validation
    dataset.set_split('val')
    classifier.eval()
    batch_generator = generate_batches(dataset, batch_size=args.batch_size, device=args.device)
    running_mean_loss = 0.0
    running_mean_acc = 0.0
    
    with torch.no_grad():
        for batch_idx, batch_dict in enumerate(batch_generator):
            y_pred = classifier(batch_dict['x'])
            loss = criterion(y_pred, batch_dict['y'])
            
            batch_loss = loss.item()
            batch_acc = compute_accuracy(y_pred, batch_dict['y'])
            running_mean_loss += (batch_loss - running_mean_loss) / (batch_idx + 1)
            running_mean_acc += (batch_acc - running_mean_acc) / (batch_idx + 1)
        
        train_state['val_loss'].append(running_mean_loss)
        train_state['val_acc'].append(running_mean_acc)
