model = LinearSVC(class_weight="balanced")

model.fit(X_train_vector, Y_train)
preds = model.predict(X_test_vector)

acc = accuracy_score(Y_test, preds)
f1 = f1_score(Y_test, preds, average="weighted")
recall = recall_score(Y_test, preds, average="weighted")
precision = precision_score(Y_test, preds, average="weighted")
print(acc)
print(precision)
print(recall)
print(f1)

score = cross_val_score(model, X_train_vector, Y_train, cv=kf, scoring="accuracy")
print(f"cross validation score: {score.mean()}")

# new_comment = ["mliha"]
# X_dev_vector = vectorizer.transform(new_comment)
# p = model.predict(X_dev_vector)
# print(p)

model = MultinomialNB(alpha=0.5)

model.fit(X_train_vector, Y_train)
preds = model.predict(X_test_vector)

acc = accuracy_score(Y_test, preds)
f1 = f1_score(Y_test, preds, average="weighted")
recall = recall_score(Y_test, preds, average="weighted")
precision = precision_score(Y_test, preds, average="weighted", zero_division=0)
print(acc)
print(precision)
print(recall)
print(f1)
score = cross_val_score(model, X_train_vector, Y_train, cv=kf, scoring="accuracy")
print(f"cross validation score: {score.mean()}")
new_comment = ["mliha"]
X_dev_vector = vectorizer.transform(new_comment)
p = model.predict(X_dev_vector)
print(p)

model = DecisionTreeClassifier(class_weight="balanced")

model.fit(X_train_vector, Y_train)
preds = model.predict(X_test_vector)

acc = accuracy_score(Y_test, preds)
f1 = f1_score(Y_test, preds, average="weighted")
recall = recall_score(Y_test, preds, average="weighted")
precision = precision_score(Y_test, preds, average="weighted")
print(acc)
print(precision)
print(recall)
print(f1)
score = cross_val_score(model, X_train_vector, Y_train, cv=kf, scoring="accuracy")
print(f"cross validation score: {score.mean()}")
new_comment = ["mliha"]
X_dev_vector = vectorizer.transform(new_comment)
p = model.predict(X_dev_vector)
print(p)