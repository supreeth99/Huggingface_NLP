from datasets import load_dataset
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
import string
import warnings
warnings.filterwarnings("ignore")
    
def clean_titles(df):
    punctuations = string.punctuation
    
    df.loc[:, 'title'] = df.title.str.replace('@USER', '') 
    #Remove mentions (@USER)
    df.loc[:, 'title'] = df.title.str.replace('URL', '') 
    #Remove URLs
    df.loc[:, 'title'] = df.title.str.replace('&amp', 'and') 
    #Replace ampersand (&) with and
    df.loc[:, 'title'] = df.title.str.replace('&lt','') #Remove &lt
    df.loc[:, 'title'] = df.title.str.replace('&gt','') #Remove &gt
    # df.loc[:, 'title'] = df.title.str.replace('\d+','') #Remove numbers
    df.loc[:, 'title'] = df.title.str.lower() #Lowercase

    #Remove punctuations
    for punctuation in punctuations:
        df.loc[:, 'title'] = df.title.str.replace(punctuation, '')

    df.loc[:, 'title'] = df.astype(str).apply(
        lambda x: x.str.encode('ascii', 'ignore').str.decode('ascii')
    )
    #Remove emojis

    df.loc[:, 'title'] = df.title.str.strip() 




def modelrun(model,tokenizer,dataset):  
    data = dataset['train']
    df_pandas = pd.DataFrame(data)
    # print(df_pandas[:10])
    # clean_titles(df_pandas)
    # print(df_pandas[:10])
    # print(type(df_pandas))
    # print(df_pandas['title'])
    text_list= list(df_pandas['title'])
    label = list(df_pandas['clickbait'])


    input_ids = [tokenizer.encode(sent, add_special_tokens=True) for sent in text_list[:300]]
    max_len = max([len(sent) for sent in input_ids])
    input_ids = [sent + [0] * (max_len - len(sent)) for sent in input_ids]
    attention_masks = []
    for sent in input_ids:
        att_mask = [int(token_id > 0) for token_id in sent]
        attention_masks.append(att_mask)

    # Convert to tensors
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    # Set up the device and batch size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bs = 32

    # Create the DataLoader
    data = list(zip(input_ids, attention_masks))

    # Run the model
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_masks)
        logits = outputs[0]
        pred = torch.argmax(logits,dim=1).tolist()
        # print(pred)
        # print(logits)
        print(classification_report(label[:300],pred))
        confusion_m = confusion_matrix(label[:300],pred)
        print(confusion_m)
        print("Accuracy score:",accuracy_score(label[:300],pred))

if __name__ == '__main__':
    dataset = load_dataset("marksverdhei/clickbait_title_classification")
    
    model1 = AutoModelForSequenceClassification.from_pretrained("valurank/distilroberta-clickbait")
    tokenizer1 = AutoTokenizer.from_pretrained("valurank/distilroberta-clickbait",do_lower_case=True)
    modelrun(model1,tokenizer1,dataset)
    
    model2 = AutoModelForSequenceClassification.from_pretrained("Stremie/roberta-base-clickbait")
    tokenizer2 = AutoTokenizer.from_pretrained("Stremie/roberta-base-clickbait")
    modelrun(model2,tokenizer2,dataset)

    model3 = AutoModelForSequenceClassification.from_pretrained("elozano/bert-base-cased-clickbait-news")
    tokenizer3 = AutoTokenizer.from_pretrained("elozano/bert-base-cased-clickbait-news")
    modelrun(model3,tokenizer3,dataset)