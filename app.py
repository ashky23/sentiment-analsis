from flask import Flask, render_template, request
import pickle

# loading classifier model
filename = 'log_model.pickle'
clf_model = pickle.load(open(filename, 'rb'))

# loading vectorizer
filename = 'vectorizer.pickle'
vectorizer = pickle.load(open(filename, 'rb'))


app = Flask(__name__)


@app.route("/")
@app.route("/home")
def home():
    return render_template('index.html')


@app.route("/analysis", methods=["POST"])
def analysis():
    input = request.form['text']
    vectorized_input = vectorizer.transform([input])
    result = clf_model.predict(vectorized_input)

    print(result)
    response = ""
    if result[0] == 0:
        response = "Negative"
    elif result[0] == 2:
        response = "Neutral"
    elif result[0] == 4:
        response = "Positive"
    return render_template('index.html', data=response)


@app.errorhandler(404)
def not_found(e):
    return render_template('404.html')


if __name__ == '__main__':
    app.run(port=5000, debug=True)
