from flask_cors import CORS
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS
model = SentenceTransformer('all-MiniLM-L6-v2')

# Mock employee data (replace with real data later)
employees = [
    {"id": 1, "name": "John", "skills": "Python JavaScript", "rating": 4.5},
    {"id": 2, "name": "Alice", "skills": "Design UI/UX", "rating": 4.2}
]

@app.route('/suggest', methods=['POST'])
def suggest():
    task_req = request.json.get("requirements", "")
    task_embedding = model.encode(task_req)
    
    suggestions = []
    for emp in employees:
        emp_embedding = model.encode(emp["skills"])
        similarity = cosine_similarity([task_embedding], [emp_embedding])[0][0]
        weighted_score = similarity * emp["rating"]
        suggestions.append({"id": emp["id"], "name": emp["name"], "score": weighted_score})
    
    suggestions.sort(key=lambda x: x["score"], reverse=True)
    return jsonify({"suggestions": suggestions[:3]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)