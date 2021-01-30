import torch
import sys
from utility import*
import pandas as pd

def testing(batch_size, test_loader, model, device):
    model.to(device)
    model.eval()
    with torch.no_grad():
        ret_output = []
        for inputs in test_loader:
            inputs = inputs.to(device,dtype=torch.long)
            outputs = model(inputs)
            outputs = outputs.squeeze()
            outputs[outputs>=0.5] = 1 # 大於等於 0.5 為負面
            outputs[outputs<0.5] = 0 # 小於 0.5 為正面
            ret_output += outputs.int().tolist()
        return ret_output

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sen_len = 20
    batch_size = 64
    w2v_path = 'w2v_all.model' 
    print("loading testing data ...")
    test_x = load_testing_data(sys.argv[1])
    #input('stop')
    preprocess = Preprocess(test_x, sen_len, w2v_path=w2v_path)
    embedding = preprocess.make_embedding(load=True)
    test_x = preprocess.sentence_word2idx()
    test_dataset = TwitterDataset(X=test_x, y=None)
    test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                                batch_size = batch_size,
                                                shuffle = False)
    print('\nload model ...')
    model = torch.load('ckpt.model')
    outputs = testing(batch_size, test_loader, model, device)
    #output csv file
    tmp = pd.DataFrame({"id":[str(i) for i in range(len(test_x))],"label":outputs})
    print("save csv ...")
    tmp.to_csv(sys.argv[2], index=False)
    print("Finish Predicting")
