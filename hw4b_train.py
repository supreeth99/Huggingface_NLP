#importing the necessary libraries
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
import pandas as pd
import string


import warnings
warnings.filterwarnings("ignore")
    
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



#specifying the bert-base-uncased model
bert_model = 'bert-base-uncased'

#create a tokenizer
tokenizer = BertTokenizer.from_pretrained(bert_model)

#create a model
model = BertForSequenceClassification.from_pretrained(bert_model, num_labels=2)

#train the model
model.train()

#specify the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)

#specify the loss function
criterion = nn.CrossEntropyLoss()

#specify the number of epochs
epochs = 5

#specify the training data

test_path = 'C:/Users/bssup/Documents/fall22/NLP/hw2/archive/olid-training-v1.0.tsv'
tweet =  pd.read_csv(test_path,sep='\t')
tweet = tweet.iloc[1:,:]
tweet.drop(['subtask_b','subtask_c'],axis=1)
tweet.replace({'NOT': 0, 'OFF': 1},inplace=True)
# print(tweet)

# tweet.drop("subtask_c",axis=1)
clean_tweets(tweet)
train_data = tweet
print(tweet)
#train the model
for epoch in range(epochs):
  #loop through each example in the training data
  for example,label in zip(train_data.tweet,train_data.subtask_a):
    # print(example)
    # print(label)
    #encode the text and label
    encoded_input = tokenizer.encode_plus(
      example,
      add_special_tokens=True,
      max_length=256,
      pad_to_max_length=True,
      return_attention_mask=True,
      return_tensors='pt'
    )
    encoded_label = torch.tensor(label)

    #pass the encoded data to the model
    outputs = model(
      input_ids=encoded_input['input_ids'],
      attention_mask=encoded_input['attention_mask'],
      labels=encoded_label
    )

    #calculate the loss
    loss = outputs[0]

    #backpropagate the gradients
    loss.backward()

    #update the weights
    optimizer.step()
    optimizer.zero_grad()

  #print the loss after each epoch
  print('Epoch {} Loss {:.4f}'.format(epoch+1, loss.item()))

#save the trained model
torch.save(model.state_dict(), 'bert-base-uncased-tweet-classifier.pt')