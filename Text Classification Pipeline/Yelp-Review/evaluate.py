import torch
import torch..nn.functional as F
def evaluate_test_data(batch_generator, classifier, compute_accuracy, criterion, train_state):
  classifier.eval()
  running_mean_loss = 0.0
  running_mean_accuracy = 0.0
  for batch_idx, batch_dict in enumerate(batch_generator):
    y_pred = classifier(batch_dict['x'])
    loss = criterion(y_pred, batch_dict['y'])
    batch_loss = loss.item()
    batch_acc = compute_accuracy(y_pred, batch_dict['y'])
    
    running_mean_loss += (batch_loss - running__mean_loss) / (batch_idx + 1)
    running_mean_accuracy += (batch_loss - running_mean_accuracy) / (batch_idx + 1)

  train_state['test_loss'] = running_mean_loss
  train_state['test_acc'] = running_mean_accuracy

  
def infer(review, classifier, vectorizer, preprocess_text, threshold=0.5):
  """ Args:
  review (str): the text review
  classifier (Model): the trained model
  vectorizer (ReviewVectorizer): the vectorizer associated with the classifier
  threshold (float): the decision threshold
  preprocess_text (function): to preprocess text review
  
  Returns:
  (str): the class of the review"""
  
  review = preprocess_text(review)
  vectorized_review = torch.tensor(vectorizer.vectorize(review))
  pred = classifier(vectorized_review.view(1, -1))
  prob = F.sigmoid(pred).item()
  if prob > threshold:
    return "negative"
  else:
    return "positive"
  
def inspect_weights(classifier, vectorizer):
  fc1_weights = classifier.fc1.weight.detach()[0] # detach to stop the expensive graph building by autograd
  _, indices = torch.sort(fc1_weights, descending=True, dim=0)
  indices = indices.numpy().tolist()
  
  print("most negative words")
  for i in range(20):
    print(vectorizer.vocab.get_token_on_idx(indices[i]))
    
  print("most positive words")
  indices.reverse()
  for i in range(20):
    print(vectorizer.vocab.get_token_on_idx(indices[i]))
    
                                         
  
