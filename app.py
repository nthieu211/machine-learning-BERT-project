from flask import Flask, render_template, request
import io
import boto3
import torch
import torch.nn as nn
from transformers import AutoTokenizer

# specify GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = {
    "model": "bert-base-uncased",
    "hidden_size": 768
}

# config tokenizer
tokenizer = AutoTokenizer.from_pretrained(config["model"], do_lower_case=True)

class Net(nn.Module):
    def __init__(self, bert):
        super(Net, self).__init__()
        self.bert = bert

        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(config["hidden_size"], 2)

    def forward(self, sent_id, masks):
        q = self.bert(sent_id, attention_mask=masks)
        cls = q[1]
        x = self.dropout(cls)
        x = self.fc(x)
        return x


class TOEICBert:
    def __init__(self, model, lr, n_epochs, train_loader):
        super(TOEICBert, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(self.device)
        self.model = model
        self.lr = lr
        self.n_epochs = n_epochs

        self.ce=torch.nn.CrossEntropyLoss()
        self.optimizer = self._get_optimizer()

    def test(self, item):
            global tokenizer
            self.model.eval()
            sents = []
            for id in range(1, 5):
                question  = item['question'].replace("___", item[str(id)])
                sents.append(question)
            inputs = tokenizer.batch_encode_plus(sents, padding=True, truncation=True, max_length=64, return_tensors='pt') 
            with torch.no_grad():
                output = self.model(inputs['input_ids'].to(device), inputs['attention_mask'].to(device))
            prediction = torch.softmax(output, dim=-1)
            
            return torch.argmax(prediction[:, 1], dim=-1).item()+1


# ========================================================================================================================

app = Flask(__name__)

s3_client = boto3.client('s3')

# Download the model from the s3 bucket
response = s3_client.get_object(Bucket='bert-toeic', Key='model_pytorch.pt')

# Read the binary data from the response
binary_data = response['Body'].read()

model = torch.load(binary_data)

@app.route("/",methods=["GET"])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    questionPart1 = request.form['questionPart1']
    questionPart2 = request.form['questionPart2']
    question = questionPart1 + ' ___ ' + questionPart2

    optionA = request.form['optionA']
    optionB = request.form['optionB']
    optionC = request.form['optionC']
    optionD = request.form['optionD']

    _quest = {'1': f"{optionA}",
    '2': f"{optionB}",
    '3': f"{optionC}",
    '4': f"{optionD}",
    'question': f"{question}"}
    result = model.test(_quest)
    answer = f"{chr(64 + int(result))}. {_quest[str(result)]}"
    return render_template('index.html', answer=answer, questionPart1=questionPart1, questionPart2=questionPart2, optionA=optionA, optionB=optionB, optionC=optionC, optionD=optionD)

if __name__ == '__main__':
    app.run(debug=True)