from keras.models import load_model
import glob
from keras.datasets import cifar10
from sklearn.metrics import accuracy_score


(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_test = x_test.astype('float32')
x_test /= 255

for model in glob.glob("*.h5"):
    print ("Loading model {0}".format(model), end = '')
    
    x = load_model(model)
    
    predictions = x.predict_classes(x_test, verbose=0)
    print(" Accuracy: {0}".format(accuracy_score(y_test.squeeze(), predictions)))
    
    