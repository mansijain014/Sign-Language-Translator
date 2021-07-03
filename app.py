from flask import Flask, Response, render_template, url_for
import cv2
import keras
# import tensorflow as tf

app = Flask(__name__)
reconstructed_model = keras.models.load_model("sign-language-translator.h5")

def getLetter(result):
    classLabels = {
        0: 'A',
        1: 'B',
        2: 'C',
        3: 'D',
        4: 'E',
        5: 'F',
        6: 'G',
        7: 'H',
        8: 'I',
        9: 'K',
        10: 'L',
        11: 'M',
        12: 'N',
        13: 'O',
        14: 'P',
        15: 'Q',
        16: 'R',
        17: 'S',
        18: 'T',
        19: 'U',
        20: 'V',
        21: 'W',
        22: 'X',
        23: 'Y'
    }
    try:
        res = int(result)
        return classLabels[res]
    except:
        return "Error"

def generate_frames():
    camera=cv2.VideoCapture(0)
    while True:
        success, frame=camera.read()
        roi = frame[100:400, 320:620]
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi = cv2.resize(roi, (28,28), interpolation = cv2.INTER_AREA)
        
        copy = frame.copy()
        cv2.rectangle(copy, (320,100), (620, 400), (255,0,0),5)
        
        roi = roi.reshape(1,28,28,1)
        
        result = str(reconstructed_model.predict_classes(roi, 1, verbose = 0)[0])
        cv2.putText(copy, getLetter(result), (300,100), cv2.FONT_HERSHEY_COMPLEX, 2,(0,255,0),2)
        cv2.imshow('Sign Language Translator', copy)

        key = cv2.waitKey(1)
        if key == ord("x"):
            break
    camera.release()
    cv2.destroyAllWindows()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/translate')    
def translate():
    return render_template('translate.html')

@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')    

@app.route('/blog1')
def blog1():
    return render_template('blog1.html')    

@app.route('/blog2')
def blog2():
    return render_template('blog2.html') 

@app.route('/blog3')
def blog3():
    return render_template('blog3.html')  

@app.route('/blog4')
def blog4():
    return render_template('blog4.html')  

@app.route('/blog5')
def blog5():
    return render_template('blog5.html')  

@app.route('/blog6')
def blog6():
    return render_template('blog6.html')                         

if __name__=='__main__':
    app.run(debug=True)    