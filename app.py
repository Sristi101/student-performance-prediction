from flask import Flask, request, render_template
import os

from src.pipeline.predict_pipeline import PredictPipeline, CustomData

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/health')
def health():
    return "ok", 200


@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method=="GET":
        return render_template("action.html")
    else:
        data=CustomData(
            gender=request.form.get("gender"),
            race_ethnicity=request.form.get("race_ethnicity"),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get("reading_score")),
            writing_score=float(request.form.get("writing_score"))
        )

        input_df=data.to_dataframe()
        print(input_df)


        predict_pipeline=PredictPipeline()
        prediction = predict_pipeline.predict(input_df)

        return render_template('action.html', pred_result=round(prediction[0], 2))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)