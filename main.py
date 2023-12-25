import pandas as pd
import numpy as np
from scipy.stats import entropy
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text


data = pd.read_csv('lung_cancer_examples.csv')
print(data)
df = pd.DataFrame(data)

#Verinin gruplandırılması

df['Age_Class'] = pd.cut(df['Age'], bins=[0, 25, 40, float('inf')], labels=['Genc', 'Orta', 'Yasli'])
df['Alkhol_Class'] = pd.cut(df['Alkhol'], bins=[-1, 3, 6, float('inf')], labels=['0', '1', '2'])
df['Smokes_Class'] = pd.cut(df['Smokes'], bins=[-1, 7, 13, float('inf')], labels=['0', '1', '2'])
df['AreaQ_Class'] = pd.cut(df['AreaQ'], bins=[-1, 3, 7, float('inf')], labels=['0', '1', '2'])

new_data= df[['Age_Class', 'Alkhol_Class','Smokes_Class','AreaQ_Class','Result']]
Entropi_data= df[['Result']]
print(new_data)

def calculate_entropy(column):
    _, counts = np.unique(column, return_counts=True)
    probabilities = counts / counts.sum()
    return entropy(probabilities, base=2)
Entropi_result=calculate_entropy(Entropi_data)
print("Result_Entropisi = ",Entropi_result)

################### Age Kazanç Hesaplanması#############################

Age0 = new_data[new_data['Age_Class'] == 'Genc']
Age1 = new_data[new_data['Age_Class'] == 'Orta']
Age2 = new_data[new_data['Age_Class'] == 'Yasli']

Age0_result= Age0[['Result']]
Age0_entropy=calculate_entropy(Age0_result)

Age1_result= Age1[['Result']]
Age1_entropy=calculate_entropy(Age1_result)

Age2_result= Age2[['Result']]
Age2_entropy=calculate_entropy(Age2_result)

kazanc_Age= Entropi_result-((len(Age0)*Age0_entropy
                             +len(Age1)*Age1_entropy
                             +len(Age2)*Age2_entropy)/(len(Age0)+len(Age1)+len(Age2)))
print("Kazanc_Age = ",kazanc_Age)

################### Smokes Kazanç Hesaplanması##########################

Smokes0 = new_data[new_data['Smokes_Class'] == '0']
Smokes1 = new_data[new_data['Smokes_Class'] == '1']
Smokes2 = new_data[new_data['Smokes_Class'] == '2']

Smokes0_result= Smokes0[['Result']]
Smokes0_entropy=calculate_entropy(Smokes0_result)

Smokes1_result= Smokes1[['Result']]
Smokes1_entropy=calculate_entropy(Smokes1_result)

Smokes2_result= Smokes2[['Result']]
Smokes2_entropy=calculate_entropy(Smokes2_result)

kazanc_Smokes= Entropi_result-((len(Smokes0)*Smokes0_entropy+
                                len(Smokes1)*Smokes1_entropy+
                                len(Smokes2)*Smokes2_entropy)/(len(Smokes0)+len(Smokes1)+len(Smokes2)))
print("Kazanc_Smoke = ",kazanc_Smokes)

################### Alkhol Kazanç Hesaplanması##########################

Alkhol0 = new_data[new_data['Alkhol_Class'] == '0']
Alkhol1 = new_data[new_data['Alkhol_Class'] == '1']
Alkhol2 = new_data[new_data['Alkhol_Class'] == '2']

Alkhol0_result= Alkhol0[['Result']]
Alkhol0_entropy=calculate_entropy(Alkhol0_result)

Alkhol1_result= Alkhol1[['Result']]
Alkhol1_entropy=calculate_entropy(Alkhol1_result)

Alkhol2_result= Alkhol2[['Result']]
Alkhol2_entropy=calculate_entropy(Alkhol2_result)

kazanc_Alkhol= Entropi_result-((len(Alkhol0)*Alkhol0_entropy+
                                len(Alkhol1)*Alkhol1_entropy+
                                len(Alkhol2)*Alkhol2_entropy)/(len(Alkhol0)+len(Alkhol1)+len(Alkhol2)))
print("Kazanc_Alkhol = ",kazanc_Alkhol)

################### AreaQ Kazanç Hesaplanması##########################

AreaQ0 = new_data[new_data['AreaQ_Class'] == '0']
AreaQ1 = new_data[new_data['AreaQ_Class'] == '1']
AreaQ2 = new_data[new_data['AreaQ_Class'] == '2']

AreaQ0_result= AreaQ0[['Result']]
AreaQ0_entropy=calculate_entropy(AreaQ0_result)

AreaQ1_result= AreaQ1[['Result']]
AreaQ1_entropy=calculate_entropy(AreaQ1_result)

AreaQ2_result= AreaQ2[['Result']]
AreaQ2_entropy=calculate_entropy(AreaQ2_result)

kazanc_AreaQ= Entropi_result-((len(AreaQ0)*AreaQ0_entropy+
                                len(AreaQ1)*AreaQ1_entropy+
                                len(AreaQ2)*AreaQ2_entropy)/(len(AreaQ0)+len(AreaQ1)+len(AreaQ2)))
print("Kazanc_AreaQ = ",kazanc_AreaQ)

##############################################################

class_mapping = {'Genc': 0, 'Orta': 1, 'Yasli': 2}
new_data['Age_Class'] = df['Age_Class'].map(class_mapping)

X = new_data[['Alkhol_Class', 'Smokes_Class', 'AreaQ_Class','Age_Class']]
y = new_data['Result']

# Eğitim ve test veri setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Karar ağacı modeli oluşturma
clf = DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
tree_rules = export_text(clf, feature_names=['Alkhol_Class', 'Smokes_Class', 'AreaQ_Class', 'Age_Class'])
print(tree_rules)

# Karar ağacından test etme
new_data_test = pd.DataFrame({'Alkhol_Class': [0], 'Smokes_Class': [2], 'AreaQ_Class': [0], 'Age_Class':[2]})
prediction = clf.predict(new_data_test)

print(f'Tahmin: {prediction[0]}')



