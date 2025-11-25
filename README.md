# 📧 Gmail Spam Classifier 

This project automatically detects spam emails in Gmail using Logistic Regression.
A dashboard lets users view emails, delete spam, export data for training, and retrain the model.

## 🚀 System Workflow

| Stage | Tool Used                        | Purpose                                      |
|-------|----------------------------------|----------------------------------------------|
| 1️⃣ Initial Training        | Logistic Regression (Scikit-Learn) | Build the main spam classifier model         |
| 2️⃣ Live Email Classification | Logistic Regression              | Predicts spam / not spam in Gmail inbox      |
| 3️⃣ Auto-Label Correction   | Hugging Face (open-source)        | Corrects labels for edge-case emails         |
| 4️⃣ Retrain Anytime         | Logistic Regression               | Uses saved CSV to update the model           |

# 🧠 Model Details

Main classifier → Logistic Regression trained on:

spam_assassin.csv

emails_training.csv (exported from Gmail dashboard)

Saved Models:
```
classifier.pkl
vectorizer.pkl
```

Hugging Face model used only to fix misclassifications:
mrm8488/bert-tiny-finetuned-sms-spam-detection

## 🛠 Tech Used

| Component       | Technology                  |
|-----------------|-----------------------------|
| Backend ML      | Python + Scikit-Learn       |
| Hugging Face    | Open-source BERT model      |
| UI Dashboard    | Gradio                      |
| Mail Access     | Gmail API (OAuth 2.0)       |


# 📌 Folder Structure
```pgsql
📂 project
 ┣ 📄 train_model.py
 ┣ 📄 automate.py
 ┣ 📄 spam_assassin.csv
 ┣ 📄 emails_training.csv (auto-created after saving)
 ┣ 📄 classifier.pkl
 ┣ 📄 vectorizer.pkl
 ┣ 📄 credentials.json
 ┣ 📄 token.json
 ┗ 📄 README.md
```
# ▶️ How to Run

# 1️⃣ Install Dependencies
```bash
pip install pandas scikit-learn transformers gradio google-api-python-client google-auth-httplib2 google-auth-oauthlib
```

# 2️⃣ Train Model First
```bash
python train_model.py
```


This generates:
```
classifier.pkl
vectorizer.pkl
```

# 3️⃣ Launch Gmail Spam Classifier Dashboard
```bash
python automate.py
```
## 📂 Dashboard Features

| Feature          | Function                                   |
|------------------|--------------------------------------------|
| 🔐 Login         | Authenticate Gmail account                 |
| 📥 Fetch Emails  | Load inbox + show spam prediction          |
| 🗑 Delete        | Move email to trash                        |
| 💾 Save to CSV   | Export (text, label) pairs for retraining  |
| ⚡ Retrain Model | Runs training again with new CSV           |

# ✔ How Label Correction Works

The system corrects uncertain Logistic Regression predictions:
Detects emails where spam probability is unclear (near decision boundary)
Sends only those to the Hugging Face spam model
Rewrites the target value appropriately
Saves final cleaned dataset into emails_training.csv
So the main model stays fast (LogReg) while Hugging Face improves accuracy.

# 🔑 How to Get credentials.json and Enable Gmail Access

To allow the project to read and delete emails from Gmail, you must configure a Google Cloud OAuth client.

## 📌 Step-by-Step Setup

1. Go to Google Cloud Console → [Google Cloud Console](https://console.cloud.google.com/)

2. **Create a New Project**

3. On the left sidebar, navigate to:  
   **APIs & Services → Library**

4. Search for **Gmail API** → Click **Enable**

5. Go to:  
   **APIs & Services → OAuth Consent Screen**
   - Set **User Type → External**
   - Fill app details (name + email) → **Save**
   - Scroll down to **Test Users → Add Users**
     - Add your Gmail address here  
     ⚠ *If you don’t add yourself as a test user, Gmail login will NOT work.*

6. Go to:  
   **APIs & Services → Credentials**
   - Click **Create Credentials → OAuth Client ID**
   - Select: **Application type → Desktop App**
   - Click **Create** and then **Download JSON**

7. Rename the downloaded file to:  
   **`credentials.json`**

8. Place the file in your project folder (same directory as `automate.py`)

🔒 Security Note

Do not share:
```
credentials.json
token.json
```
These files give Gmail access.


