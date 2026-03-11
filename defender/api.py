from flask import Flask, request, jsonify
import defence

app = Flask(__name__)

# POST route
@app.route('/start_defence', methods=['POST'])
def submit():
    data = request.get_json()

    name = data.get('model_name')
    dataset = data.get('dataset')

    defence.initialize_defence_pipeline(name)

    return jsonify({
        "status": "started",
        "model": name,
        "dataset": dataset
    }), 200


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)