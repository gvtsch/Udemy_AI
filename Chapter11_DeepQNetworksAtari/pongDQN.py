import tensorflow as tf
from tensorflow.keras.layers import Activation, Dense, Input, Conv2D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import Huber
import numpy as np

class DQN(tf.keras.Model): # Erben von einer Klasse
    def __init__(self, img_shape, num_actions, learning_rate):
        super().__init__()  # Sicherstellen, dass alles, was von der Base-Class geerbt wird, 
                            # auch initialisiert wird, mit dessen Konstruktor
        self.img_shape = img_shape
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.loss = Huber()
        self.optimizer = RMSprop(
            learning_rate=0.00025,
            rho=0.95,
            epsilon=0.01
        )
        self.internal_model = self.build_model()

    def build_model(self):
        input_img = Input(shape=self.img_shape)
        # Hyperparameter und Netz-Struktur laut Nature-Journal
        x = Conv2D(filters=32, kernel_size=8, strides=4, padding="same")(input_img) 
        x = Activation("relu")(x)
        x = Conv2D(filters=64, kernel_size=4, strides=2, padding="same")(x) 
        x = Activation("relu")(x)
        x = Conv2D(filters=64, kernel_size=3, strides=1, padding="same")(x) 
        x = Activation("relu")(x)
        x = Flatten()(x) # Zu Vektor casten
        x = Dense(units=256)(x)
        x = Activation("relu")(x)
        q_value_pred = Dense(self.num_actions)(x) 
        model = Model(
            inputs=input_img,
            outputs=q_value_pred
        )
        model.compile(
            loss=self.loss,
            optimizer=self.optimizer
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

if __name__=="__main__":
    dqn = DQN(
        img_shape=(84, 84, 4), # 4 --> 4 Frames für Richtungserkennung
        num_actions=2,
        learning_rate=0.001
    )
    dqn.internal_model.summary()