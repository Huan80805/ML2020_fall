from gensim.models import word2vec
from utility import*
import spacy
    
def train_word2vec(x):
    # 訓練 word to vector 的 word embedding
    model = word2vec.Word2Vec(x, size=250, window=5, min_count=5, workers=12, iter=10, sg=1)
    return model

def token_extract(token):
    return not(token.is_punct or token.is_stop)
    
if __name__ == "__main__":
    #load training data
    NLP = spacy.load('en_core_web_lg')
    with open('training_label.txt', 'r',encoding="utf-8") as f:
        lines = f.readlines()
        lines = [line.strip('\n').split(' ') for line in lines]
        train_y = [line[0] for line in lines]
        train_x = [' '.join(line[2:]) for line in lines]
        print('stripped')
        train_x = [[token.lemma_ for token in NLP(line) if token_extract(token)] for line in train_x]
    print('training_data_loaded')
    with open('testing_data.txt','r',encoding='utf-8') as f:
        lines = f.readlines()
        lines = [line.strip('\n')]
        test_x = ["".join(line.strip('\n').split(",")[1:]) for line in lines[1:]]
        test_x = [[token.lemma_ for token in NLP(line) if token_extract(token)] for line in test_x]   
    
    model = train_word2vec(train_x + test_x)
    print("saving model ...")
    model.save('w2v_spacy.model')