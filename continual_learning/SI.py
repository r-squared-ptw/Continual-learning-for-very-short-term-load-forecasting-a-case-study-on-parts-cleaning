# Implementation based on: https://www.sciencedirect.com/science/article/pii/S2405896320313823

import numpy as np
import tensorflow as tf
import copy
from tensorflow.keras import Model

from eta_ml_lib.basics.utils import log

# CustomModel class to override parts of the tf training loop (small omega needs to be updated after every epoch).
class CustomModel(Model):
    def __init__(self,  *args, **kwargs):
        super(CustomModel, self).__init__(*args, **kwargs)

        # tracks how much a parameter contributes to a change in loss
        self.small_omega_var = {}
        
        # saves the weights of the previous step during training
        self.previous_weights_during_training = {}
        
        # saves the importances of the parameters in previous tasks
        self.big_omega_var = {}
        
        # saves the weights of the previous task
        self.previous_weights = {}
    
    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape_penalty:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            loss, penalty = self.loss(y_true=y, y_pred=y_pred)
            penalty_loss = loss + penalty
        gradients_with_penalty = tape_penalty.gradient(penalty_loss, self.trainable_variables)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients_with_penalty, self.trainable_variables))

        for grad_penalty, var in zip(gradients_with_penalty, self.trainable_variables):
            if grad_penalty is not None:
                # Compute the update for small_omega_var
                update = -grad_penalty * (var - self.previous_weights_during_training[var.name])
                self.small_omega_var[var.name].assign(self.small_omega_var[var.name] + update)
                self.previous_weights_during_training[var.name].assign(var)

        # Update metrics (includes the metric that tracks the loss)
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

class SI:
    """
    Class for Synaptic Intelligence to tackle Concept Drift.
    The algorithm introduces a penalty during training. based on the importance of parameters observed in previous tasks, to make sure the updated models performance on previous tasks won't suffer performance losses.
    """
    def __init__(self):
        self.si_model = None
        self.strategy = tf.distribute.MirroredStrategy()

    def __getstate__(self):
        # Exclude the strategy from the pickling process
        state = self.__dict__.copy()
        del state['strategy']
        return state

    def __setstate__(self, state):
        # Restore the strategy upon unpickling
        self.__dict__.update(state)
        self.strategy = tf.distribute.MirroredStrategy()

    def retrain(self, model, epochs, loss_fn, optimizer, train_set, test_set=None, c=1, xi=0.1):
        """
        Retrains the model to adapt to the new task presented in train_set.

        :param model: The model to be retrained.
        :param epochs: The number of epochs to train for
        :param loss_fn: The loss function to use for training.
        :param optimizer: The optimizer to use for training.
        :param train_set: The Training Set holding data describing the concept drift.
        :param test_set: A Test Set that can be used during training for validation. 
        :param c: Regularization parameter. The higher the more the model is penalized for changing parameters that were important for previous tasks. Heavily dependent on the range of output values that are predicted.
        :param xi: Damping parameter, to help cases where the difference between current weights and prior weights is close to 0.
        """

        # loss function calculations are based on the formulars of: https://www.sciencedirect.com/science/article/pii/S2405896320313823
        def si_loss_fn(y_true, y_pred):
            loss = loss_fn(y_true, y_pred)
            penalty = 0.
            for weight in self.si_model.trainable_weights:
                if weight.name in self.si_model.big_omega_var.keys():
                    penalty += tf.reduce_sum(tf.multiply(self.si_model.big_omega_var[weight.name], tf.square(weight - self.si_model.previous_weights[weight.name])))
            return loss, c * penalty
        
        log.info("Starting retraining....")
        # Compile model to use new loss function but keep original weights
        with self.strategy.scope():
            if self.si_model is None:
                new_model = tf.keras.models.clone_model(model)
                new_model.set_weights(model.get_weights())
                self.si_model = CustomModel(inputs=new_model.inputs, outputs=new_model.outputs)
            
                for weight in self.si_model.weights:
                    self.si_model.small_omega_var[weight.name] = tf.Variable(tf.zeros(weight.shape), trainable=False)
                    self.si_model.previous_weights_during_training[weight.name] = tf.Variable(copy.deepcopy(weight), trainable=False)
                    self.si_model.big_omega_var[weight.name] = tf.Variable(tf.zeros(weight.shape), trainable=False)
                    self.si_model.previous_weights[weight.name] = tf.Variable(copy.deepcopy(weight), trainable=False)

            self.si_model.compile(optimizer=optimizer, loss=si_loss_fn)
        log.info("Fitting model....")
        self.si_model.fit(train_set[0], train_set[1], epochs=epochs)

        log.info("Calculating parameter importance....")
        with self.strategy.scope():
        # After each task is complete, update big_omega and reset small_omega and store weights
            for weight in self.si_model.trainable_weights:
                self.si_model.big_omega_var[weight.name].assign_add(tf.divide(self.si_model.small_omega_var[weight.name], (xi + tf.square(weight-self.si_model.previous_weights_during_training[weight.name]))))
                self.si_model.small_omega_var[weight.name].assign(self.si_model.small_omega_var[weight.name] * 0.0)
                self.si_model.previous_weights_during_training[weight.name] = tf.Variable(copy.deepcopy(weight), trainable=False)
                self.si_model.previous_weights[weight.name] = tf.Variable(copy.deepcopy(weight), trainable=False)

        if test_set is not None:
            log.info("Test-Set Loss: " + str(loss_fn(self.si_model.predict(test_set[0]), test_set[1])))

        # return a copy of the model so that it can be used independently from the class
        return_model = tf.keras.models.clone_model(self.si_model)
        return_model.set_weights(copy.deepcopy(self.si_model.trainable_weights))

        return return_model