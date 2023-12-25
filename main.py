import pandas as pd
import numpy as np
from scipy.stats import entropy

data = pd.read_csv('lung_cancer_examples.csv')
print(data)
df = pd.DataFrame(data)

#Verinin gruplandırılması

df['Age_Class'] = pd.cut(df['Age'], bins=[0, 25, 40, float('inf')], labels=['Genc', 'Orta', 'Yaslı'])
df['Alkhol_Class'] = pd.cut(df['Alkhol'], bins=[-1, 3, 6, float('inf')], labels=['0', '1', '2'])
df['Smokes_Class'] = pd.cut(df['Smokes'], bins=[-1, 7, 13, float('inf')], labels=['0', '1', '2'])
df['AreaQ_Class'] = pd.cut(df['AreaQ'], bins=[-1, 3, 7, float('inf')], labels=['0', '1', '2'])

new_data= df[['Age_Class', 'Alkhol_Class','Smokes_Class','AreaQ_Class','Result']]
Entropi_data= df[['Age_Class','Alkhol_Class','Smokes_Class','AreaQ_Class','Result']]
print(new_data)

def calculate_entropy(column):
    _, counts = np.unique(column, return_counts=True)
    probabilities = counts / counts.sum()
    return entropy(probabilities, base=2)

# Her bir sütun için entropiyi hesapla
for column in Entropi_data.columns:
    column_entropy = calculate_entropy(Entropi_data[column])
    print(f'Entropy for {column}: {column_entropy}')
x0 = new_data[new_data['Age_Class'] == 'Genc']
x1 = new_data[new_data['Age_Class'] == 'Orta']
x2 = new_data[new_data['Age_Class'] == 'Yaslı']


print(x0,x1,x2)
x0_result= x0[['Result']]
xx0=calculate_entropy(x0_result)
print(xx0,len(x0))

x1_result= x1[['Result']]
xx1=calculate_entropy(x1_result)
print(xx1,len(x1))

x2_result= x2[['Result']]
xx2=calculate_entropy(x1_result)
print(xx2,len(x2))

print(x2)