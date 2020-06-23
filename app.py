import requests
from flask import Flask, render_template
from final import model_train

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('train.html')

@app.route('/train')
def train():
    model_train()
    print(requests.get('http://15.206.165.107/train').content)
    return "Trained"

if __name__ == "__main__":
    app.run(debug=True)