import json,sys
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import HashingVectorizer
if sys.version_info[0]>=3: raw_input=input
transformer=HashingVectorizer(stop_words='english')
trn=[]
trn_lbl=[]
f=open('training.json')
for i in range(int(f.readline())):
    h=json.loads(f.readline())
    trn.append(h['question']+"\r\n"+h['excerpt'])
    trn_lbl.append(h['topic'])
f.close()
train = transformer.fit_transform(trn)
svm=LinearSVC()
svm.fit(train,trn_lbl)

_test=[]
for i in range(int(raw_input())):
    h=json.loads(raw_input())
    _test.append(h['question']+"\r\n"+h['excerpt'])
test = transformer.transform(_test)
test_label=svm.predict(test)
for e in test_label: print(e)