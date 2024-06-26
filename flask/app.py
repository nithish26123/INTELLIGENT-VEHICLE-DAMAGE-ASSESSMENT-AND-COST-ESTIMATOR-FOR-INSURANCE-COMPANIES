
import numpy as np
import os

from flask import Flask, app, request, render_template
from keras import models
from keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.python.ops.gen_array_ops import concat
from keras.applications.inception_v3 import preprocess_input

from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename


model1=load_model("C:/Users/sivabala pc/Desktop/Vehicle-Damage-Assement-and-Cost-Estimator-main/venv/body.h5")
model2=load_model("C:/Users/sivabala pc/Desktop/Vehicle-Damage-Assement-and-Cost-Estimator-main/venv/level.h5")

    
app = Flask(__name__)



@app.route('/')
def index():
    return render_template('index.html')




@app.route('/login')
def login():
    return render_template('login.html')



@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/prediction')
def prediction():
    return render_template('prediction.html')




@app.route('/afterreg', methods=['POST'])
def afterreg():
    x = [x for x in request.form.values()]
    print(x)
    data = {
        '_id': x[1],
        'name': x[0],
        'psw': x[2]
    }
    print(data)

    query = {'_id': {'$eq': data['_id']}}

    docs = my_database.get_query_result(query)
    print(docs)

    print(len(docs.all()))

    if (len(docs.all()) == 0):
        url = my_database.create_document(data)
        response = request.get(url)
        return render_template('login.html', pred="Registration Successful, Please login using your details")
    else:
        return render_template('register.html', pred="You are already a member, Please login using your details")





@app.route('/afterlogin', methods=['POST'])
def afterlogin():
    user = request.form['_id']
    passw = request.form['psw']
    print(user, passw)

    query = {'_id': {'$eq': user}}

    docs = my_database.get_query_result(query)
    print(docs)

    print(len(docs.all()))

    if (len(docs.all()) == 0):
        return render_template('login.html', pred="The Username is not found")
    else:
        if ((user == docs[0][0]['_id'] and passw == docs[0][0]['psw'])):
            return redirect(url_for('prediction'))
        else:
            print('Invalid User')


@app.route('/logout')
def logout():
    return render_template('logout.html')



@app.route('/result', methods=['POST'])
def result():
    if request.method == "POST":
        f = request.files['image']
        basepath = os.path.dirname(__file__)
        filepath = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(filepath)

        img = image.load_img(filepath, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        print(img)
       
        
        prediction1 = np.argmax(model1.predict(x))
        prediction2 = np.argmax(model2.predict(x))

        index1 = ['00-front', '01-rear', '02-side']
        index2 = ['01-minor', '02-moderate', '03-severe']

        res1 = index1[prediction1]
        res2 = index2[prediction2]
        result1=format(str(res1))
        result2=format(str(res2))
     
        if (result1 == "00-front" and result2 == "01-minor"):
            value = "3000 - 5000 INR"
        elif (result1 == "00-front" and result2 == "02-moderate"):
            value = "6000 - 8000 INR"
        elif (result1 == "00-front" and result2 == "03-severe"):
            value = "9000 - 11000 INR"
        elif (result1 == "01-rear" and result2 == "01-minor"):
            value = "4000 - 6000 INR"
        elif (result1 == "01-rear" and result2 == "02-moderate"):
            value = "7000 - 9000 INR"
        elif (result1 == "01-rear" and result2 == "03-severe"):
            value = "11000 - 13000 INR"
        elif (result1 == "02-side" and result2 == "01-minor"):
            value = "6000 - 8000 INR"
        elif (result1 == "02-side" and result2 == "02-moderate"):
            value = "9000 - 11000 INR"
        elif (result1 == "02-side" and result2 == "03-severe"):
            value = "12000 - 15000 INR"
        else:
            value = "16000 - 50000 INR"
        print(value)
    
        return render_template('prediction.html',prediction=value,img_path=img)



if __name__ == "__main__":
    app.run(debug=False,host = '0.0.0.0')
