from flask import Flask, render_template, request,redirect
from werkzeug.utils import secure_filename
import os


crackResult = {}
crackResultFromcam = {}
app = Flask(__name__)
UPLOAD_FOLDER = 'images/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



@app.route("/dict")
def saveResult(dictionary= None):
    dict_result = dictionary
    return dict_result
    


@app.route("/")
def home():
    return  render_template("index.html")


@app.route("/upload")
def upload():
    return render_template("home.html")
    
@app.route("/register",methods=['POST'])
def createuser():
    username = request.form['username']
    phone = request.form['phonenumber']
    email = request.form['email']
    password = request.form['password']

    try:
        import mysql.connector

        try:
            mysqlConnection =  mysql.connector.connect(host = "localhost", user = "root", password = "password",database = "UserDetails")
            mysqlCursor = mysqlConnection.cursor()
            mysqlCursor.execute("insert into userRegister values(%s,%s,%s,%s)",(username,phone,email,password))
            mysqlConnection.commit()
            return render_template("index.html",result1 = "Register Successfully")

        except:
            print("database connection error ! ")            


    except:
        print('import mysql connector error !') 
    return render_template("index.html", result2 = "Registration Failed!")


@app.route("/login",methods = ['POST'])
def userlogin():
    username1 = request.form['username']
    password1 = request.form['password']
    try:
        import mysql.connector

        try:
            mysqlConnection =  mysql.connector.connect(host = "localhost", user = "root", password = "password",database = "UserDetails")
            mysqlCursor = mysqlConnection.cursor()
            mysqlCursor.execute("select username, password from userRegister where username = %s and password =  %s",(username1,password1))
            result = mysqlCursor.fetchall()
            if (username1 =="" and password1==""):
                pass
            if(result[0][0] == username1 and result[0][1] == password1):
                return render_template("welcome.html")
            # else:
            #     jsloginerror = """alert('login error!')"""
            #     js2py.eval(jsloginerror)   

        except:
            print("database connection error ! ")            


    except:
        print('import mysql connector error !')
        
    return render_template("index.html",result = "please enter correct username and password")




    #return render_template("welcome.html")

@app.route('/predict', methods= ['POST'])
def predict():
    import tensorflow as tf
    from tensorflow.keras.preprocessing import image
    import numpy as np
    from tensorflow.keras.applications.resnet50 import preprocess_input

    f = request.files['files']
    if f.filename != '':
        for f in request.files.getlist('files'):
            f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))
    class_name = ['crack not found','crack found']
    loaded_model = tf.keras.models.load_model("./model/crack_detection_model.h5")

    for list_images in os.listdir("/home/akshay/flask/images"):
        img = image.load_img("./images/"+list_images, target_size=(120, 120))
        img_array = image.img_to_array(img)
        img_batch = np.expand_dims(img_array, axis=0)
        img_preprocessed = preprocess_input(img_batch)
        y_pred = np.squeeze((loaded_model.predict(img_preprocessed) >= 0.5).astype(np.int))
        crackResult[list_images] = class_name[y_pred]

    saveResult(crackResult)
    
    return render_template("result.html",result = crackResult)


@app.route('/cam', methods = ['POST'])
def cam():
    import pickle
    import cv2
    import numpy as np
    model = cv2.dnn.readNetFromCaffe("/home/akshay/Desktop/keras-caffe-converter/src/conversion/output/tt/generated_prototxt.prototxt","/home/akshay/Desktop/keras-caffe-converter/src/conversion/output/tt/crack_detection_new.caffemodel")
    i = 0
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 1)
    noCrackcount = 0
    scrackcount = 0
    while(True):
        ret , frame = cap.read()
        img = cv2.resize(frame , (120, 120))
        img_blob = cv2.dnn.blobFromImage(img)
        model.setInput(img_blob)
        output = model.forward()
        result = np.argmin(output)
        if(result >= 5):
            noCrackcount = noCrackcount+1
        else:
            scrackcount = scrackcount+1

        if(scrackcount == 60):
            crackResultFromcam[str(i)+".jpg"] = "crack found"
            cv2.imwrite("./imageFromCamera/"+str(i)+".jpg",frame)
            i = i+1
            scrackcount = 0
        if(noCrackcount == 60):
            crackResultFromcam[str(i)+".jpg"] = "crack not found"
            # cv2.imwrite("./imageFromCamera/"+str(i)+".jpg",frame)
            i=i+1
            noCrackcount = 0
        cv2.imshow("",frame)
        if(cv2.waitKey(1) & 0xFF == ord('q')):
            break
    saveResult(crackResultFromcam)
    cap.release()
    cv2.destroyAllWindows()


    return render_template("result.html",result = crackResultFromcam)

@app.route("/pie", methods= ['GET'])
def pie():
    # savecrakValue =  saveResult()
    crack_count = 0
    not_crack_count = 0
    for value in crackResult.values():
        if (value == "crack found"):
            crack_count = crack_count+1
        else:
            not_crack_count = not_crack_count+1
    dictionary= {"crack found":crack_count,"crack not found":not_crack_count}
    # dictionary = {"crack found":10,"crack not found":10}
    return render_template("pie.html",data = dictionary)
@app.route('/piecam')
def piecam():
    crack_count = 0
    not_crack_count = 0
    for value in crackResultFromcam.values():
        if (value == "crack found"):
            crack_count = crack_count+1
        else:
            not_crack_count = not_crack_count+1
    dictionary= {"crack found":crack_count,"crack not found":not_crack_count}
    return render_template("pie.html", data = dictionary)

@app.route("/forgetpassword")
def forget():
    return render_template("forgetpassword.html")

if __name__ == "__main__":
    app.run(debug=True)
