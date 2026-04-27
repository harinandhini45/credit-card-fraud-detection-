from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    report = classification_report(y_test, preds)

    return acc, cm, report
