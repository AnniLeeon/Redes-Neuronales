import network
import mnist_loader
import pickle
import Cross


training_data, validation_data , test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)
validation_data = list(validation_data)
test_data = list(test_data)

Red = Cross.Network([784, 30, 10])
Red.SGD(training_data, 30, 10, 4.0,  test_data = test_data )

Archivo = open('archivo_prueba.pkl', 'wb')
pickle.dump(Red, Archivo)
Archivo.close()
exit()




