#construct dictionary for BOW
import numpy as np
import spacy
    
def token_extract(token):
    return not(token.is_punct or token.is_stop)
    
if __name__ == "__main__":
    NLP = spacy.load('en_core_web_sm')
    dictionary = set()
    with open('training_label.txt', 'r',encoding="utf-8") as f:
        lines = f.readlines()
        lines = [' '.join(line.strip('\n').split(' ')[2:]) for line in lines]
        print('-----------loading training data---------------')
        for i,line in enumerate(lines):
            for token in NLP(line):
                if token_extract(token):
                    dictionary.add(token.text)
            print('sentence count #{}'.format(i+1), end='\r')
    with open('testing_data.txt','r',encoding='utf-8') as f:
        lines = f.readlines()
        lines = ["".join(line.strip('\n').split(",")[1:]) for line in lines[1:]]
        print('\n-----------loading testing data---------------')
        for i,line in enumerate(lines):
            for token in NLP(line):
                if token_extract(token):
                    dictionary.add(token.text)
            print('sentence count #{}'.format(i+1), end='\r')
    print('\ntotal_words:{}\n'.format(len(dictionary)))
    dictionary = np.array(list(dictionary))
    np.save('dictionary.npy',dictionary)
    print('dictionary saved')