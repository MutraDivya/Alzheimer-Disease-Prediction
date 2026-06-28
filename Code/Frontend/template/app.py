from flask import Flask, render_template, request, redirect, url_for, flash
import mysql.connector
import io
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import resnet50, swin_t, ResNet50_Weights, Swin_T_Weights
import base64

app = Flask(__name__)

# MySQL connection setup
mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    port="3306",
    database='Alzheimer'
)
mycursor = mydb.cursor()

#MySQL query functions
def executionquery(query, values):
    mycursor.execute(query, values)
    mydb.commit()

def retrivequery1(query, values):
    mycursor.execute(query, values)
    return mycursor.fetchall()

def retrivequery2(query):
    mycursor.execute(query)
    return mycursor.fetchall()

# === Route function for index ===
@app.route('/')
def index():
    
    return render_template('index.html')




@app.route('/register', methods=["GET", "POST"])
def register():
        if request.method == "POST":
            name = request.form.get('name')
            email = request.form['email']
            password = request.form['password']
            confirmpassword = request.form['confirmpassword']
            if password == confirmpassword:
                query = "SELECT UPPER(email) FROM users"
                email_data = retrivequery2(query)
                email_data_list = []
                for i in email_data:
                    email_data_list.append(i[0])
                if email.upper() not in email_data_list:
                    query = "INSERT INTO users (name, email, password) VALUES (%s, %s, %s)"
                    values = (name, email, password)
                    executionquery(query, values)

                    return render_template('login.html', message="Successfully Registered!")
                return render_template('register.html', message="This email ID is already exists!")
            return render_template('register.html', message="Confirm password is not match!")
        return render_template('register.html')


@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form['email']
        password = request.form['password']
        
        query = "SELECT UPPER(email) FROM users"
        email_data = retrivequery2(query)
        email_data_list = []
        for i in email_data:
            email_data_list.append(i[0])

        if email.upper() in email_data_list:
            query = "SELECT UPPER(password) FROM users WHERE email = %s"
            values = (email,)
            password__data = retrivequery1(query, values)
            if password.upper() == password__data[0][0]:
                global user_email
                user_email = email

                return redirect("/home")
            return render_template('login.html', message= "Invalid Password!!")
        return render_template('login.html', message= "This email ID does not exist!")
    return render_template('login.html')


@app.route('/about', methods=["GET", "POST"])
def about():

        return render_template('about.html')

# ==============================================================
# 1. MODEL DEFINITION (ResNet-50 + Swin-T hybrid)
# ==============================================================
class ResSwinNet(nn.Module):
    def __init__(self, num_classes):
        super(ResSwinNet, self).__init__()
        # Pre-trained backbones
        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.swin   = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)

        # Remove original heads
        self.resnet.fc = nn.Identity()
        self.swin.head = nn.Identity()

        res_dim  = 2048
        swin_dim = 768

        # Fusion + classifier
        self.fc = nn.Sequential(
            nn.Linear(res_dim + swin_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        res_feat = self.resnet(x)
        swin_feat = self.swin(x)
        combined = torch.cat((res_feat, swin_feat), dim=1)
        return self.fc(combined)

# ==============================================================
# 2. GLOBAL SETUP (run once at import time)
# ==============================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 5

model = ResSwinNet(num_classes=num_classes).to(device)

# ---- Load the best checkpoint (saved by your training script) ----
MODEL_PATH = 'ResSwin_final.pth'               # <-- put the file in the project root
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ---- Same transform as training/validation ----
infer_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225])
])

# ---- Class names – order MUST match ImageFolder order ----
CLASS_NAMES = [
    'Irrelavent',          # 0
    'Mild Impairment',     # 1
    'Moderate Impairment', # 2
    'No Impairment',       # 3
    'Very Mild Impairment' # 4
]

ALLOWED_EXT = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT

# ==============================================================
# 3. ROUTE: /home
# ==============================================================
@app.route('/home', methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        return render_template('home.html',
                               prediction=None,
                               confidence=None,
                               image_data_url=None)

    if 'mriImage' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['mriImage']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    if not allowed_file(file.filename):
        flash('Allowed image types: png, jpg, jpeg')
        return redirect(request.url)

    try:
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        # Encode image to base64 for HTML
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        image_data_url = f"data:image/jpeg;base64,{img_base64}"

        # Preprocess for model
        tensor = infer_transform(img).unsqueeze(0).to(device)

        # Inference
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0)
            confidence, pred_idx = torch.max(probs, dim=0)
            confidence = round(confidence.item() * 100, 1)
            pred_idx = pred_idx.item()

        predicted_class = CLASS_NAMES[pred_idx]

        return render_template('home.html',
                               prediction=predicted_class,
                               confidence=confidence,
                               image_data_url=image_data_url)

    except Exception as e:
        flash(f'Error processing image: {str(e)}')
        return redirect(request.url)



if __name__ == '__main__':
    print(f"Server running on http://127.0.0.1:5000")
    app.run(debug=True)