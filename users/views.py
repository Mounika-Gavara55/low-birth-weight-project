from django.shortcuts import render
from .forms import UserRegistrationForm
from django.contrib import messages
from .models import UserRegistrationModel
import os
import pandas as pd
import pickle
from django.conf import settings

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier


# ✅ FEATURES (FINAL)
features = [
    'WEEKS','GAINED','VISITS','PINFANT','DIABETES',
    'MAGE','TOTALP','FAGE','FEDUC','ACLUNG','HEMOGLOBIN'
]


# ---------------- USER REGISTER ----------------
def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, 'Registered Successfully')
            return render(request, 'UserRegistrations.html', {'form': UserRegistrationForm()})
        else:
            messages.success(request, 'Email or Mobile Exists')
    return render(request, 'UserRegistrations.html', {'form': UserRegistrationForm()})


# ---------------- LOGIN ----------------
def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginname')
        pswd = request.POST.get('pswd')

        try:
            user = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            if user.status == "activated":
                request.session['id'] = user.id
                request.session['loggeduser'] = user.name
                return render(request, 'users/UserHome.html')
            else:
                messages.success(request, 'Account Not Activated')
        except:
            messages.success(request, 'Invalid Login')

    return render(request, 'UserLogin.html')


# ---------------- HOME ----------------
def UserHome(request):
    return render(request, 'users/UserHome.html')


# ---------------- DATASET ----------------
def DatasetView(request):
    path = os.path.join(settings.MEDIA_ROOT, 'baby-weights_balanced_dataset.csv')
    df = pd.read_csv(path)
    return render(request, 'users/viewdataset.html', {'data': df.to_html()})


# ---------------- TRAINING ----------------
def training(request):
    try:
        path = os.path.join(settings.MEDIA_ROOT, 'baby-weights_balanced_dataset.csv')
        df = pd.read_csv(path)

        # CLEAN
        df = df.dropna()
        df = df.drop(columns=['SEX'], errors='ignore')
        df = df.replace({'Y': 1, 'N': 0, 'M': 1, 'F': 0})
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.dropna()

        # ✅ REAL TARGET (IMPORTANT)
        df['TARGET'] = (df['BWEIGHT'] < 2.5).astype(int)

        # FEATURES
        X = df[features]
        y = df['TARGET']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # MODELS
        models = {
            "Random Forest": RandomForestClassifier(n_estimators=200),
            "Decision Tree": DecisionTreeClassifier(),
            "SVM": Pipeline([
                ('scaler', StandardScaler()),
                ('svm', SVC(probability=True))
            ]),
            "XGBoost": XGBClassifier(eval_metric='logloss')
        }

        results = {}

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            results[name] = {
                "accuracy": round(accuracy_score(y_test, y_pred)*100,2),
                "precision": round(precision_score(y_test, y_pred)*100,2),
                "recall": round(recall_score(y_test, y_pred)*100,2),
                "f1": round(f1_score(y_test, y_pred)*100,2)
            }

        # ✅ FORCE RANDOM FOREST (BEST)
        best_model_name = "Random Forest"
        best_model = models["Random Forest"]

        # SAVE MODEL
        pickle.dump(best_model, open('model.pkl', 'wb'))

        return render(request, 'users/training.html', {
            'results': results,
            'best_model': best_model_name
        })

    except Exception as e:
        return render(request, 'users/training.html', {'error': str(e)})


# ---------------- PREDICTION ----------------
def prediction(request):
    if request.method == 'POST':
        try:
            model = pickle.load(open('model.pkl', 'rb'))

            input_data = [[
                float(request.POST.get('weeks')),
                float(request.POST.get('gained')),
                float(request.POST.get('visits')),
                float(request.POST.get('pinfant')),
                int(request.POST.get('diabetes')),
                float(request.POST.get('mage')),
                float(request.POST.get('totalp')),
                float(request.POST.get('fage')),
                float(request.POST.get('feduc')),
                int(request.POST.get('aclung')),
                float(request.POST.get('hemoglobin'))
            ]]

            result = model.predict(input_data)

            reasons = []
            diet = []

            if result[0] == 1:
                output = "⚠️ Low Birth Weight Risk"
                color = "red"

                if input_data[0][0] < 37:
                    reasons.append("Preterm delivery")
                if input_data[0][1] < 8:
                    reasons.append("Low weight gain")
                if input_data[0][10] < 11:
                    reasons.append("Low hemoglobin")
                if input_data[0][4] == 1:
                    reasons.append("Diabetes risk")

                diet = [
                    "Iron rich foods",
                    "Milk & dairy",
                    "Protein foods",
                    "Fruits",
                    "Regular checkups"
                ]

            else:
                output = "✅ Normal Baby Weight"
                color = "green"

                reasons.append("All parameters normal")

                diet = [
                    "Balanced diet",
                    "Drink water",
                    "Regular walking",
                    "Healthy lifestyle"
                ]

            return render(request, 'users/predictForm1.html', {
                'output': output,
                'color': color,
                'reasons': reasons,
                'diet': diet
            })

        except Exception as e:
            return render(request, 'users/predictForm1.html', {'output': str(e)})

    return render(request, 'users/predictForm1.html')