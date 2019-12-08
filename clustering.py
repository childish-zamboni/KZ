from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from nltk.tokenize import sent_tokenize
from flask import Flask
from flask import render_template
from flask import send_from_directory
from flask import request
from flask_ngrok import run_with_ngrok

#text = "In the late 1990s, a lonely teenager on the West Coast fired up his dial-up modem to find someone to talk to. He was a shy kid, too introverted to feel fully comfortable in the real world, and he logged on to the early internet’s bare-bones web forums for a sense of connection. There he found friends: other people who were awkward in real life, particularly when it came to sex and dating."

vectorizer = TfidfVectorizer(stop_words='english')

def train_and_predict(text, text2):
    documents = sent_tokenize(text)

    X = vectorizer.fit_transform(documents)

    true_k = 3
    model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
    model.fit(X)

    clusters = ""

    print("Top terms per cluster:")
    order_centroids = model.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names()
    for i in range(true_k):
        clusters += "Cluster %d: " % i + "["
        print("Cluster %d:" % i),
        for ind in order_centroids[i, :4]:
            clusters += ' %s' % terms[ind]
            print(' %s' % terms[ind]),
            print
        clusters += " ]\n"

    Y = vectorizer.transform([text2])
    prediction = model.predict(Y)
    print(prediction)
    return("Your action was most closely associated with the following cluster: " + str(prediction[0])+ "\n\nBelow are the top clusters of related events that were identified:\n" + clusters)

#train_and_predict(text, "I love cake")

app = Flask(__name__)
#run_with_ngrok(app)

@app.route("/")
def hello():
    return render_template("index.html")

@app.route('/js/<path:path>')
def send_js(path):
    return send_from_directory('public/js', path)

@app.route('/css/<path:path>')
def send_css(path):
    return send_from_directory('public/css', path)

@app.route("/predict")
def predict():
    log = request.args.get("log")
    action = request.args.get("action")
    return (train_and_predict(log, action))

if __name__ == "__main__":
    app.run()
