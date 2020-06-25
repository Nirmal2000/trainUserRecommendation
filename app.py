import requests
from flask import Flask, render_template
from final import model_train
from multiprocessing import Process

app = Flask(__name__)
app.config['ENV']='development'
app.config['DEBUG']=True
trainProcess = Process(target=model_train,daemon=True)

@app.route('/')
def home():
    if trainProcess.is_alive() == False:
        return render_template('train.html')
    else:
        return "Model is training please wait.."

@app.route('/train')
def train():
    if trainProcess.is_alive() == False:
        trainProcess.start()
        return "Started training..."
    else:
        return "Model is training please wait..."

if __name__ == "__main__":
    app.run(debug=True)
