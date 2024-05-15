from flask import Flask, request, render_template
import pickle
import numpy as np
# load the model
with open('../../models/knn.pkl', 'rb') as file:
    model = pickle.load(file)

with open('../../models/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# create a flask application
app = Flask(__name__)


@app.route("/", methods=["GET"])
def root():
    # read the file contents and send them to client
    return render_template('index2.html')


@app.route("/classify", methods=["POST"])
def classify():
    # get the values entered by user
    print(request.form)
    gender = int(request.form.get("gender"))
    own_car = int(request.form.get("own_car"))
    own_reality = int(request.form.get("own_reality"))
    cnt_children = int(request.form.get("cnt_children"))
    income = float(request.form.get("income"))
    education_type = int(request.form.get("education_type"))
    family_status = int(request.form.get("family_status"))
    days_birth = float(request.form.get("days_birth")) * -1
    personal_phone = float(request.form.get("personal_phone"))
    days_employeed = float(request.form.get("days_employeed")) * -1
    work_phone = float(request.form.get("work_phone"))
    email = float(request.form.get("email"))
    begin_months = float(request.form.get("begin_months")) * -1
    job = int(request.form.get("job"))


    scaled_features = scaler.transform([[income, days_birth, days_employeed, begin_months]])
    # print(scaled_features)
    # return f"{scaled_features.flatten()}"

    features = np.array(
        [gender, own_car, own_reality, cnt_children, education_type, family_status, work_phone, personal_phone, email,
         job])
    combined_features = np.concatenate((features, scaled_features.flatten()))
    answers = model.predict([combined_features])
    # li = []
    # li.append(combined_features)
    # return f"{li}"

    if int(answers[0]) == 0:
        return f"Credit Card Approve"
    else:
        return "Not Approve"



# start the application
app.run(host="0.0.0.0", port=8000, debug=True)
