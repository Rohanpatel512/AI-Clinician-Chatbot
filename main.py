import os 
from flask import Flask, render_template, request, jsonify
from rag_pipeline import rag_pipeline

root = os.pathname.file(__file__)
app = Flask(__name__, template_folder=root)

@app.route('/')
def index():
    return render_template('frontend/index.html')

@app.route('/chat', methods=["POST"])
def chat():

    data = request.get_json()
    user_query = data['message']
    history = data['history']

    response = rag_pipeline(user_query, history)

    return jsonify({'res': response})

