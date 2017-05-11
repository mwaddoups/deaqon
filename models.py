import numpy as np

from sklearn.linear_model import SGDRegressor, SGDClassifier

from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.optimizers import Adam

class BaseModel:
    def init(self, n_obs, n_act):
        pass
    
    def predict(self, data):
        pass

    def fit(self, X, y):
        pass


class SGDRegModel(BaseModel):
    def __init__(self, **kwargs):
        self.models = []
        self.model_kwargs = kwargs
        
    def init(self, n_obs, n_act):
        self.models = []
        for i in xrange(n_act):
            model = SGDRegressor(**self.model_kwargs)
            model.partial_fit(np.random.rand(1, n_obs), 
                              np.random.rand(1))
            self.models.append(model)
            
    def predict(self, data):
        predictions = np.zeros((data.shape[0], len(self.models)))
        for i, model in enumerate(self.models):
            predictions[:, i] = model.predict(data)
        return predictions
    
    def fit(self, X, y):
        for i, model in enumerate(self.models):
            model.partial_fit(X, y[:, i].ravel())

class SGDClfModel(SGDRegModel):
    def init(self, n_obs, n_act): 
        self.models = []
        for i in xrange(n_act):
            model = SGDClassifier(**self.model_kwargs)
            
            model.partial_fit(np.random.rand(1, n_obs), [0], classes=[0, 1])
            self.models.append(model)

    def predict(self, data):
        predictions = np.zeros((data.shape[0], len(self.models)))
        for i, model in enumerate(self.models):
            predictions[:, i] = model.predict_proba(data)[:, 1]
        predictions /= np.sum(predictions, axis=1).reshape(-1, 1)
        return predictions

class NNRegModel(BaseModel):    
    def __init__(self, hidden_layers, optimizer=None):
        if optimizer is None:
            optimizer = Adam()
        self.hidden_layers = hidden_layers
        self.optimizer = optimizer

    def init(self, n_obs, n_act):
        inputs = Input(shape=(n_obs,))

        hidden = inputs
        for num_nodes, dropout in self.hidden_layers:
            hidden = Dense(num_nodes, activation='relu')(hidden)
            if dropout > 0:
                hidden = Dropout(dropout)(hidden)

        outputs = Dense(n_act, activation='linear')(hidden)
        model = Model(inputs=inputs, outputs=outputs)
        
        model.compile(loss='mse', optimizer=self.optimizer)
        
        self.model = model
            
    def predict(self, data):
        return self.model.predict(data, verbose=0)
    
    def fit(self, X, y):
        self.model.train_on_batch(X, y)

class NNClfModel(NNRegModel):
    
    def init(self, n_obs, n_act):
        inputs = Input(shape=(n_obs,))

        hidden = inputs
        for num_nodes, dropout in self.hidden_layers:
            hidden = Dense(num_nodes, activation='relu')(hidden)
            if dropout > 0:
                hidden = Dropout(dropout)(hidden)

        outputs = Dense(n_act, activation='softmax')(hidden)
        model = Model(inputs=inputs, outputs=outputs)
        
        model.compile(loss='categorical_crossentropy', optimizer=self.optimizer)
        
        self.model = model
