import pickle
import pandas as pd
import gradio as gr
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import os
from transformers import pipeline

SCOPES = ['https://www.googleapis.com/auth/gmail.modify']

clf = pickle.load(open("classifier.pkl", "rb"))
vec = pickle.load(open("vectorizer.pkl", "rb"))
hf_classifier = pipeline("text-classification", model="mrm8488/bert-tiny-finetuned-sms-spam-detection", return_all_scores=False)

service = None
last_df = None

def gmail_login():
    global service
    creds = None
    if os.path.exists('token.json'):
        from google.oauth2.credentials import Credentials
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    service = build('gmail', 'v1', credentials=creds)
    return "Login Successful ✅"

def fetch_emails(max_emails=20):
    global last_df
    results = service.users().messages().list(userId='me', maxResults=max_emails).execute()
    messages = results.get('messages', [])
    rows = []
    for msg in messages:
        txt = service.users().messages().get(userId='me', id=msg['id']).execute()
        headers = txt['payload'].get('headers', [])
        subject = sender = ""
        for h in headers:
            if h['name'] == 'Subject':
                subject = h['value']
            if h['name'] == 'From':
                sender = h['value']
        snippet = txt.get('snippet', '')
        cleaned = subject + " " + snippet
        pred = clf.predict(vec.transform([cleaned]))[0]
        label = "🚫 Spam" if pred == 1 else "✅ Not Spam"
        rows.append([msg['id'], sender, subject, snippet, label, "Delete"])
    df = pd.DataFrame(rows, columns=["ID","From","Subject","Snippet","Prediction","Action"])
    last_df = df.copy()
    return style_df(df)

def style_df(df):
    def highlight(val):
        if val == "🚫 Spam":
            return "color: red; font-weight: bold;"
        elif val == "✅ Not Spam":
            return "color: green; font-weight: bold;"
        return ""
    return df.style.applymap(highlight, subset=["Prediction"])

def delete_email(selected_row):
    global last_df
    if selected_row is None or selected_row.empty:
        return last_df
    mid = selected_row["ID"].iloc[0]
    try:
        service.users().messages().trash(userId='me', id=mid).execute()
    except Exception as e:
        print(f"Failed to delete {mid}: {e}")
    last_df = last_df[last_df["ID"] != mid]
    return last_df

def fetch_emails_for_csv(max_emails=1000):
    results = service.users().messages().list(userId='me', maxResults=max_emails).execute()
    messages = results.get('messages', [])
    data = []
    for msg in messages:
        txt = service.users().messages().get(userId='me', id=msg['id']).execute()
        headers = txt['payload'].get('headers', [])
        subject = ""
        for h in headers:
            if h['name'] == 'Subject':
                subject = h['value']
        snippet = txt.get('snippet', '')
        cleaned = subject + " " + snippet
        pred = clf.predict(vec.transform([cleaned]))[0]
        data.append([cleaned, pred])
    df = pd.DataFrame(data, columns=["text", "target"])
    return df

def correct_labels_with_hf(df, threshold=0.6, batch_size=32):
    texts_to_check = []
    indices = []

    for i, row in df.iterrows():
        prob = clf.predict_proba(vec.transform([row["text"]]))[0][1]
        if 1-threshold < prob < threshold:
            texts_to_check.append(row["text"])
            indices.append(i)

    corrected = df["target"].tolist()
    for start in range(0, len(texts_to_check), batch_size):
        batch = texts_to_check[start:start+batch_size]
        results = hf_classifier(batch)
        for j, res in enumerate(results):
            idx = indices[start + j]
            corrected[idx] = 1 if res['label'].lower() == "spam" else 0

    df["target"] = corrected
    return df



def save_emails_for_training():
    df = fetch_emails_for_csv(max_emails=1000)
    df = correct_labels_with_hf(df)
    df.to_csv("emails_training.csv", index=False)
    return f"Saved {len(df)} emails to emails_training.csv"

def train_model():
    os.system("python train_model.py")
    return "Training completed ✅"

def ui():
    with gr.Blocks() as demo:
        gr.Markdown("## 📧 Gmail Spam Classifier Dashboard")
        login_btn = gr.Button("🔐 Login to Gmail")
        status = gr.Textbox(label="Status")
        fetch_btn = gr.Button("📥 Fetch Emails")
        table = gr.DataFrame(headers=["ID","From","Subject","Snippet","Prediction","Action"], interactive=True)
        save_btn = gr.Button("💾 Save Emails to CSV")
        save_status = gr.Textbox(label="Save Status")
        train_btn = gr.Button("⚡ Train Model")
        train_status = gr.Textbox(label="Training Status")

        login_btn.click(gmail_login, outputs=status)
        fetch_btn.click(fetch_emails, outputs=table)
        table.select(delete_email, inputs=table, outputs=table)
        save_btn.click(save_emails_for_training, outputs=save_status)
        train_btn.click(train_model, outputs=train_status)

    demo.launch()

ui()
