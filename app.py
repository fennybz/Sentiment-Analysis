
from OOPS import Model
from flask import Flask,abort,jsonify,request
import pickle

model = pickle.load(open('finalized_model_SVM.sav', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))
app=Flask(__name__)

# @app.route('/api',methods=['POST','GET'])
# def make_predict():
#         data=request.get_json(force=True)
#         sentence = request.form['q']
#         mymodel = Model()
#         result = mymodel.testing_unseen(sentence,model,tfidf)
#         #returning the output as a json
#         print(result)
#         return jsonify(result)

output = {}

def sentiment(sentence):

    mymodel = Model()
    result = mymodel.testing_unseen(sentence,model,tfidf)
    return result

@app.route("/", methods = ["GET","POST"])
def sentimentRequest():
    if request.method == "POST":
        sentence = request.form['q']
    else:
        sentence = request.args.get('q')
    sent = sentiment(sentence)
    print(sentence)
    output['sentiment'] = sent
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)