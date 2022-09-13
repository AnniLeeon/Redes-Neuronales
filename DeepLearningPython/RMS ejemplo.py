import RMS
import mnist_loader
import pickle


training_data, validation_data , test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)
validation_data = list(validation_data)
test_data = list(test_data)


Red = RMS.Network([784, 30, 10])
Red.RMS(training_data, 30, 10, 4.0, 0.5, 0.01, test_data = test_data )

Archivo = open('archivo_prueba.pkl', 'wb')
pickle.dump(Red, Archivo)
Archivo.close()
exit()




