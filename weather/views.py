
from rest_framework.decorators import api_view
from rest_framework.response import Response

import numpy as np
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Create your views here.


def predictWeatherFunction(dataFrame):
    model = load_model('C:\\Users\\kunwo\\OneDrive\\Desktop\\WeatherB\\WeatherPrediction\\Weather Forecasting\\model76.h5')
    scalar=MinMaxScaler()
    a2=scalar.fit_transform(dataFrame)
    ans1=model.predict(a2)
    ans2=scalar.inverse_transform(ans1)
    return ans2

@api_view(['POST'])  
def index(request):
    data= request.data['values']
    print(data)
    print(type(data))
    data=np.array(data) 
    data.shape=(1,6)
    return Response(predictWeatherFunction(data))
    #Precipitation , Relative humidity, Surface pressure , MAX temp ,MIN_temp  ,Wind speed at 10M 
