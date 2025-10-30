from flask import Flask, request, jsonify, render_template
from joblib import load
from scipy.sparse import hstack

app = Flask(__name__)

# Load models and vectorizers
tfidf_xgb = load("tfidf_vectorizer_xgb.joblib")   # for priority model
tfidf_svm = load("tfidf_vectorizer_svm.joblib")   # for task classifier
xgb = load("xgb_priority.joblib")                 # XGBoost model
svm = load("model_svm.joblib")                    # SVM model

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json() or request.form
    title = data.get("title", "")
    description = data.get("description", "")
    workload_hours = float(data.get("workload_hours", 0))
    completed_tasks = int(data.get("assignee_completed_tasks", 0))
    avg_completion_days = float(data.get("assignee_avg_completion_days", 0))
    days_to_due = int(data.get("days_to_due", 0))

    # --- Priority Prediction (XGBoost) ---
    text_xgb = (title + ". " + description).lower()
    text_vec_xgb = tfidf_xgb.transform([text_xgb])
    num_feat = [[workload_hours, completed_tasks, avg_completion_days, days_to_due]]
    X_priority = hstack([text_vec_xgb, num_feat])
    priority = int(xgb.predict(X_priority.toarray())[0])

    # --- Task Classification (SVM) ---
    text_svm = (title + ". " + description).lower()
    text_vec_svm = tfidf_svm.transform([text_svm])
    task_class = int(svm.predict(text_vec_svm)[0])

    result = {"priority": priority, "task_class": task_class}
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
