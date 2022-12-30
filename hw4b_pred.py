
# Import necessary libraries
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import pandas as pd
import string
 

def clean_tweets(df):
    
    punctuations = string.punctuation
    
    df.loc[:, 'tweet'] = df.tweet.str.replace('@USER', '') 
    #Remove mentions (@USER)
    df.loc[:, 'tweet'] = df.tweet.str.replace('URL', '') 
    #Remove URLs
    df.loc[:, 'tweet'] = df.tweet.str.replace('&amp', 'and') 
    #Replace ampersand (&) with and
    df.loc[:, 'tweet'] = df.tweet.str.replace('&lt','') #Remove &lt
    df.loc[:, 'tweet'] = df.tweet.str.replace('&gt','') #Remove &gt
    df.loc[:, 'tweet'] = df.tweet.str.replace('\d+','') #Remove numbers
    df.loc[:, 'tweet'] = df.tweet.str.lower() #Lowercase

    #Remove punctuations
    for punctuation in punctuations:
        df.loc[:, 'tweet'] = df.tweet.str.replace(punctuation, '')

    df.loc[:, 'tweet'] = df.astype(str).apply(
        lambda x: x.str.encode('ascii', 'ignore').str.decode('ascii')
    )
    #Remove emojis

    df.loc[:, 'tweet'] = df.tweet.str.strip() 

# Load the pre-trained model
model = BertModel.from_pretrained('bert-base-uncased-tweet-classifier.pt')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased-tweet-classifier.pt')

# Define the classifier
class tweetClassifier(nn.Module):
  def __init__(self, n_classes):
    super(tweetClassifier, self).__init__()
    self.bert = model
    self.drop = nn.Dropout(p=0.3)
    self.out = nn.Linear(768, n_classes)
  
  def forward(self, ids, mask, token_type_ids):
    _, o2 = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
    bo = self.drop(o2)
    output = self.out(bo)
    return output

# Initialize the classifier
classifier = tweetClassifier(2)

# Define a list of tweets
tweet =  pd.read_csv('C:/Users/bssup/Documents/fall22/NLP/hw2/archive/testset-levela.tsv',sep='\t')
tweet = tweet.iloc[1:,:]
tweet.drop("id",axis=1)
clean_tweets(tweet)
label = pd.read_csv('C:/Users/bssup/Documents/fall22/NLP/hw2/archive/labels-levela.csv')
tweets = tweet['tweet']
# print(tweets)



# Prepare the inputs
input_ids = []
attention_masks = []

for tweet in tweets:
  encoded_dict = tokenizer.encode_plus(
      tweet,                     
      add_special_tokens = True, 
      max_length = 64,          
      pad_to_max_length = True,
      return_attention_mask = True,  
      return_tensors = 'pt',    
  )
  input_ids.append(encoded_dict['input_ids'])
  attention_masks.append(encoded_dict['attention_mask'])

# Convert the lists into tensors
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)

# Predict the labels
with torch.no_grad():
  logits = classifier(input_ids, attention_masks)

# Get the predicted labels
predicted_labels = torch.argmax(logits, dim=1).numpy()

# Print the predicted labels
print(predicted_labels)