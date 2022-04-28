import tensorflow as tf
from tensorflow.keras.layers import Activation, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np

class DQN(tf.keras.Model): # Erben von einer Klasse
    def __init__(self, state_shape, num_actions, learning_rate):
        super().__init__()  # Sicherstellen, dass alles, was von der Base-Class geerbt wird, 
                            # auch initialisiert wird, mit dessen Konstruktor
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.internal_model = self.build_model()

    def build_model(self):
        input_state = Input(shape=self.state_shape)
        x = Dense(units=24)(input_state)
        x = Activation("relu")(x)
        x = Dense(units=24)(x)
        x = Activation("relu")(x)
        q_value_pred = Dense(self.num_actions)(x) 
        model = Model(
            inputs=input_state,
            outputs=q_value_pred
        )
        model.compile(
            loss="mse",
            optimizer=Adam(learning_rate=self.learning_rate)
        )
        return model

    def call(self, inputs):
        # Soll model.predict darstellen
        return self.internal_model(inputs).numpy() # So schneller, als predict-Methode

    def fit(self, states, q_values):
        self.internal_model.fit(
            x=states,
            y=q_values,
            verbose=0 # Keine Ausgabe in der Konsole
        )

    def update_model(self, other_model):
        self.internal_model.set_weights(other_model.get_weights())

    def load_model(self, path):
        self.internal_model.load_weights(path)

    def save_model(self, path):
        self.internal_model.save_weights(path)