import pandas as pd
import string
import re
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text


# Label output
def output_label(n):
    return "Fake News" if n == 0 else "Not A Fake News"


# Manual testing function
def manual_testing(news):
    testing_news = {"text": [news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test['text'] = new_def_test["text"].apply(clean_text)
    new_xv_test = vectorizer.transform(new_def_test["text"])

    pred_LR = LR.predict(new_xv_test)
    pred_DT = DT.predict(new_xv_test)
    pred_GB = GB.predict(new_xv_test)
    pred_RF = RF.predict(new_xv_test)

    print("\nManual Testing Results:")
    print(f"Logistic Regression: {output_label(pred_LR[0])}")
    print(f"Decision Tree: {output_label(pred_DT[0])}")
    print(f"Gradient Boosting: {output_label(pred_GB[0])}")
    print(f"Random Forest: {output_label(pred_RF[0])}")


# Main process
if __name__ == "__main__":
    # Load data
    data_fake = pd.read_csv('Fake.csv')
    data_true = pd.read_csv('True.csv')

    data_fake["class"] = 0
    data_true["class"] = 1

    data = pd.concat([data_fake, data_true], axis=0)
    data = data.drop(['title', 'subject', 'date'], axis=1)
    data = data.sample(frac=1).reset_index(drop=True)

    data['text'] = data['text'].apply(clean_text)

    # Feature extraction
    x = data['text']
    y = data['class']

    vectorizer = TfidfVectorizer(max_features=3000)
    xv = vectorizer.fit_transform(x)

    # Split
    x_train, x_test, y_train, y_test = train_test_split(xv, y, test_size=0.25, random_state=42)

    # Model training
    print("Training models...")
    NB = MultinomialNB()
    LR = LogisticRegression(max_iter=1000)
    DT = DecisionTreeClassifier()
    GB = GradientBoostingClassifier()
    RF = RandomForestClassifier()

    NB.fit(x_train, y_train)
    LR.fit(x_train, y_train)
    DT.fit(x_train, y_train)
    GB.fit(x_train, y_train)
    RF.fit(x_train, y_train)

    # Evaluation
    models = {
        "Naive Bayes": NB,
        "Logistic Regression": LR,
        "Decision Tree": DT,
        "Gradient Boosting": GB,
        "Random Forest": RF
    }

    for name, model in models.items():
        pred = model.predict(x_test)
        print(f"\n{name} Report:")
        print(f"Accuracy: {accuracy_score(y_test, pred):.4f}")
        print(classification_report(y_test, pred))

    # Save models (optional)
    joblib.dump(LR, "logistic_model.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")

    # Manual test input
    news = str(input("\nEnter a news article for testing: "))
    manual_testing(news)
    # # Example usage
    # news = "Breaking news: A fake incident occurred in the city, causing panic among the residents."