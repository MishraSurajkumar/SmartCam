import cv2
import os
from flask import Flask,request,render_template,redirect, url_for
from datetime import date
from datetime import datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
import csv

#### Defining Flask App
app = Flask(__name__)


#### Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")


#### Initializing VideoCapture object to access WebCam
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


#### If these directories don't exist, create them
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv','w') as f:
        f.write('Name,EMP ID,In Time,Out Time')


#### get a number of total registered users
def totalreg():
    return len(os.listdir('static/faces'))


#### extract the face from an image
def extract_faces(img):
    try:
        if img.shape!=(0,0,0):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            face_points = face_detector.detectMultiScale(gray, 1.3, 5)
            return face_points
        else:
            return []
    except:
        return []
    
#### Identify face using ML model
def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    if not model:
        return 'Unknown'
    return model.predict(facearray)[0]


#### A function which trains the model on all the faces available in faces folder
def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces,labels)
    joblib.dump(knn,'static/face_recognition_model.pkl')


#### Extract info from today's attendance file in attendance folder
def extract_attendance(selecteddate):
    if selecteddate:
        df = pd.read_csv(f'Attendance/Attendance-{selecteddate}.csv')
    else:
        df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    names = df['Name']
    IDs = df['EMP ID']
    times = df['In Time']
    otimes = df['Out Time']
    l = len(df)
    return names,IDs,times,l,otimes


#### Add Attendance of a specific user
def add_attendance(name):
    username = name.split('_')[0]
    empid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")
    
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    if int(empid) not in list(df['EMP ID']):
        with open(f'Attendance/Attendance-{datetoday}.csv','a') as f:
            f.write(f'\n{username},{empid},{current_time},NA')
    else:
        with open(f'Attendance/Attendance-{datetoday}.csv', 'r+', newline='') as file:
            reader = csv.DictReader(file)
            rows = list(reader)
            for row in rows:
                if row['EMP ID'] == empid:
                    row['Out Time'] = current_time
                    break
            
            file.seek(0)  # Move the file pointer to the beginning
            writer = csv.DictWriter(file, fieldnames=reader.fieldnames)
            writer.writeheader()
            writer.writerows(rows)

def getallusers():
    userlist = os.listdir('static/faces')
    names = []
    IDs = []
    l = len(userlist)

    for i in userlist:
        name,IDs = i.split('_')
        names.append(name)
        IDs.append(IDs)
    
    return userlist,names,IDs,l

def deletefolder(duser):
    pics = os.listdir(duser)
    for i in pics:
        os.remove(duser+'/'+i)
    os.rmdir(duser)

################## ROUTING FUNCTIONS #########################

#### Our main page
@app.route('/')
def home():
    names,IDs,times,l,otimes = extract_attendance('')    
    return render_template('Home.html',names=names,IDs=IDs,times=times,otimes=otimes,l=l,totalreg=totalreg(),datetoday2=datetoday2)  

#### This function will run when we click on Take Attendance Button
@app.route('/start',methods=['GET'])
def start():
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('Home.html',totalreg=totalreg(),datetoday2=datetoday2,mess='There is no trained model in the static folder. Please add a new face to continue.') 

    ret = True
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        
        faces = extract_faces(frame)  # Modify this to return a list of all detected faces
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
            face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))
            if identified_person != "Unknown":
                add_attendance(identified_person)
            cv2.putText(frame,  identified_person[0 : identified_person.find("_")], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        
        cv2.imshow('Attendance', frame)       
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    names,IDs,times,l,otimes = extract_attendance('')    
    return render_template('Home.html',names=names,IDs=IDs,times=times,otimes=otimes,l=l,totalreg=totalreg(),datetoday2=datetoday2) 


#### This function will run when we add a new user
@app.route('/add',methods=['GET','POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userimagefolder = 'static/faces/'+newusername+'_'+str(newuserid)
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    
    isNewUserFound = False
    i,j = 0,0
    cap = cv2.VideoCapture(0)
    while 1:
        _,frame = cap.read()
        faces = extract_faces(frame)
        identified_person = "Unknown"
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x, y), (x+w, y+h), (255, 0, 20), 2)
            
            # Checking if user is already exist
            if totalreg() > 0:
                face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
                identified_person = identify_face(face.reshape(1, -1))
            
            if identified_person != "Unknown":
                cv2.putText(frame,  identified_person[0 : identified_person.find("_")], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                cv2.putText(frame,f'Alert! : Your are alredy registerd with the system.',(30,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(20, 0, 240),2,cv2.LINE_AA)

            else:
                cv2.putText(frame,f'Images Captured: {i}/50',(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 20),2,cv2.LINE_AA)
                if j%10==0:
                    name = newusername+'_'+str(i)+'.jpg'
                    cv2.imwrite(userimagefolder+'/'+name,frame[y:y+h,x:x+w])
                    i+=1
                    isNewUserFound = True
            j+=1
        if j==500:
            break
        cv2.imshow('Adding new User',frame)
        if cv2.waitKey(1)==27:
            break
    cap.release()
    cv2.destroyAllWindows()
    print('Training Model')
    if not isNewUserFound:
        deletefolder(userimagefolder)
    else:
        train_model()
    names,IDs,times,l,otimes = extract_attendance('')    
    return render_template('Home.html',names=names,IDs=IDs,times=times,otimes=otimes,l=l,totalreg=totalreg(),datetoday2=datetoday2) 

#### This function will returns the list attandace for selected date
@app.route('/attendance')
def getUsersbasedonDate():
    selected_date = request.args.get('date')
    year, month, day = selected_date.split('-')

    formatted_date = f"{month}_{day}_{year[2:]}"
    names,IDs,times,l,otimes = extract_attendance(formatted_date)    
    return render_template('Home.html',names=names,IDs=IDs,times=times,otimes=otimes,l=l,totalreg=totalreg(),datetoday2=datetoday2)  


#### Our main function which runs the Flask App
if __name__ == '__main__':
    app.run(debug=True)