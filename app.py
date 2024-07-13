from flask import Flask, request, jsonify
import joblib
import re

# Flask 애플리케이션 생성
app = Flask(__name__)

# 저장된 모델 및 벡터화기 로드
svm_model = joblib.load('./model/svm_model.pkl')
vectorizer = joblib.load('./model/vectorizer.pkl')

# 데이터 전처리 함수
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z0-9 ]', '', text)
    return text

@app.route('/kr-bad-word-predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')

    # 텍스트 전처리
    preprocessed_text = preprocess_text(text)

    # 벡터화
    text_vectorized = vectorizer.transform([preprocessed_text])

    # 예측
    prediction = svm_model.predict(text_vectorized)
    is_profanity = bool(prediction[0])

    return jsonify({'text': text, 'is_profanity': is_profanity})

if __name__ == '__main__':
    app.run(debug=True)

