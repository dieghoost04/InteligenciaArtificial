import numpy as np  
import os
import matplotlib.pyplot as plt  

# Variable en la que se almacenaran los errores
__errors__ = []

# Dataset que contiene dos columnas [ YearsExperience, Salary ]
archivo_csv = 'Salary_dataset.csv'
samples_g = []
y = []

params = [2,0.3]

# Almacenamos las columnas en dos listas
with open(archivo_csv, 'r') as file:
    lines = file.readlines()
    for line in lines[1:]:  
        columns = line.strip().split(',')
        samples_g.append(float(columns[1]))  
        y.append(float(columns[2]))  

samples = samples_g.copy()
        
# Multivariable
#samples = [[1,1,1],[2,2,2],[3,3,3]]
#samples = [[1,2,3],[4,5,6],[7,8,9],[1,1,1]]
#y = [60, 55, 50,50]
#params = [.3,2,1]

# Univariable
#params = [2,0.3]
#samples = [59,44,51,42]
#y = [60,55,50,66]

print("Samples: ",samples)

for i in range(len(samples)):
    if isinstance(samples[i], list):
        samples[i]=  [1]+samples[i]
    else:
        samples[i]=  [1,samples[i]]



def GDD(params, samples, y, alpha = 0.01):
    '''
    Función de Gradient Descent.
    '''

    samples = np.array(samples)
    params = np.array(params)
    y = np.array(y)
    aux = params.copy()

    error = (params * samples)
    #print("\nMatrix con las multiplicacion de parametros individuales y = tx1 + tx2 + ... : ",error) 
    error = error.sum(axis = 1)
    #print("\nMatrix con la suma de las multiplicacion de parametros individuales y = tx1 + tx2 + ... = : ",error) 
    error = error - y
    #print("\nMatrix con la resta de cada uno de estos valores menos el real: ",error) 
    acum = error * samples.T
    #print(samples.T)
    #print("\nMatrix con la multiplicación de los errores por su sample: ",error) 
    acum = acum.sum(axis = 1)
    #print("\nMatrix con la suma de la multiplicación de los errores por su parametro: ",error)
    aux = params - alpha*(1/len(samples))*acum
    print("\nParametros viejos: ",params)
    print("\nParametros nuevos: ",aux)
    return aux


def errors(params, samples, y):
    '''
    Función que calcula los errores.
    '''
    
    samples = np.array(samples)
    params = np.array(params)
    y = np.array(y)
    aux = params.copy()
    
    error = (params * samples)
    #print("\nMatrix con las multiplicacion de parametros individuales y = tx1 + tx2 + ... : ",error) 
    error = error.sum(axis = 1)   
    #print("\nMatrix con la suma de las multiplicacion de parametros individuales y = tx1 + tx2 + ... = : ",error) 
    error = error - y
    #print("\nMatrix con la resta de cada uno de estos valores menos el real: ",error) 
    error = error ** 2
    #print("\nError al cuadrado: ",error)
    error = error.sum()/2*len(samples)
    #print(error)
    __errors__.append(error)
    return error

epochs = 0
while epochs < 100:
    print(params)
    errors(params, samples, y)
    params = GDD(params, samples, y)
    epochs += 1

h = params * samples
h = h.sum(axis = 1)

plt.plot(__errors__, color = 'b')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.show()

plt.plot(samples_g, y, marker='o', linestyle='-', color='b')
plt.plot(samples_g, h, marker='o', linestyle='-', color='r')
plt.xlabel('Years Experience')
plt.ylabel('Salary')
plt.title('Linear Regression Model vs Real Data')
plt.show()