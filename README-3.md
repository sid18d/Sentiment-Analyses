
# Sentiment Analysis
[![MIT License](https://img.shields.io/apm/l/atomic-design-ui.svg?)](https://github.com/tterb/atomic-design-ui/blob/master/LICENSEs)

# Appendix

Overview

Documentation

Installation

Deployment

Acknowledgement
## Overview

 In this project we use Hugging Face Transformers and BERT Neural Network to calculate sentiment. We also use BeautifulSoup to scrape reviews from an online food delivery website and calculate sentiment on the reviews 

## Documentation

[Pytorch](https://pytorch.org)     

[Transformers](https://huggingface.co/docs/transformers/index)   

[Beautifulsoup](https://pypi.org/project/beautifulsoup4/)   

 
## Installation

To install the packages you can execute the following command:-


Install Pytorch[**P.S. - Visit [website](https://pytorch.org)   to  install as per  configuration
]

```bash
  !pip install scikit-learn
```
 


Install Transformers

```bash
  !pip install transformers requests beautifulsoup4 pandas numpy

 ```
        

## Deployment

For Fake News Detection place your News Title in news_headline

Input :
```bash
r = requests.get('put the website   ')
soup = BeautifulSoup(r.text, 'html.parser')
regex = re.compile('.*comment.*')
results = soup.find_all('p', {'class':regex})     
reviews = [result.text for result in results]
# Extracting text from Reviews from the website using BeautifulSoup

```
```bash
def sentiment_score(review):
    tokens = tokenizer.encode(review, return_tensors='pt')
    result = model(tokens)
    return int(torch.argmax(result.logits))+1

``````
 
```bash
sentiment_score(df['review'].iloc[1])
``````

 
 Output :

 ```bash
 5
 # Sentiment score

```

## Acknowledgements

 - [Pypi](https://pypi.org)
 - [StackOverflow](https://stackoverflow.com)
  
