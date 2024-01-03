from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertModel
import torch

app = Flask(__name__)

# flask run --host=0.0.0.0

# BERT 모델과 토크나이저 초기화
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def bert_embedding(text):
    # 텍스트를 토큰화하고 BERT 입력 형식으로 변환
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)

    # 임베딩 생성
    with torch.no_grad():
        outputs = model(**inputs)

    # 마지막 은닉층의 출력을 임베딩으로 사용
    embeddings = outputs.last_hidden_state.mean(1).squeeze().tolist()
    return embeddings

@app.route('/embeddings', methods=['POST'])
def get_embeddings():
    data = request.json
    embeddings = {
        'embedding': bert_embedding(data['text']),
    }
    return jsonify(embeddings)

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 5000, debug=True)

