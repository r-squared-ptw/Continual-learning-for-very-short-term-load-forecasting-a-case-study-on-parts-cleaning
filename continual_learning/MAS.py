# Implementation based on: https://arxiv.org/abs/1711.09601
# and: https://www.sciencedirect.com/science/article/pii/S2212827123008260
import numpy as np
import tensorflow as tf
import pandas as pd
import copy
import random
from tensorflow.keras import Model

from eta_ml_lib.basics.utils import log

# CustomModel class to override parts of the tf training loop.
class CustomModel(Model):
    def __init__(self,  *args, **kwargs):
        super(CustomModel, self).__init__(*args, **kwargs)

        # saves the importances of the parameters in previous tasks
        self.big_omega_var = {}

        # collects param importances during training
        self.small_omega_var = {}
        
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

        # Update metrics (includes the metric that tracks the loss)
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

class MAS:
    """
    Class for Memory Aware Synapses to tackle Concept Drift.
    The algorithm introduces a penalty during training, based on the importance of parameters observed in previous tasks, to make sure the updated models performance on previous tasks won't suffer performance losses.
    """
    def __init__(self, parameter_importance_via_gradient=True):
        """
        :param parameter_importance_via_gradient: the method the parameter importance is approximated. True: gradients of training set are accumulated. False: a perturbated model is created to create the gradients.
        """
        self.mas_model = None
        self.parameter_importance_via_gradient = parameter_importance_via_gradient
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
    
    def retrain(self, model, epochs, loss_fn, optimizer, train_set, test_set=None, lambda_=1, delta=0.0001, frac=1.):
        """
        Retrains the model to adapt to the new task presented in train_set.

        :param model: The model to be retrained.
        :param epochs: The number of epochs to train for
        :param loss_fn: The loss function to use for training.
        :param optimizer: The optimizer to use for training.
        :param train_set: The Training Set holding data describing the concept drift.
        :param test_set: A Test Set that can be used during training for validation. 
        :param _lambda: Regularization parameter. The higher the more the model is penalized for changing parameters that were important for previous tasks. Heavily dependent on the range of output values that are predicted.
        :param delta: Perturbation parameter to estimate parameter importance. 
        :param frac: The fraction of samples that should be used for importance matrix estimation. Use a small value for faster computation (0 < frac <= 1).
        """
        def mas_loss_fn(y_true, y_pred):
            loss = loss_fn(y_true, y_pred)
            penalty = 0.
            for weight in self.mas_model.trainable_weights:
                if weight.name in self.mas_model.big_omega_var.keys():
                    penalty += tf.reduce_sum(tf.multiply(self.mas_model.big_omega_var[weight.name], tf.square(weight - self.mas_model.previous_weights[weight.name])))
            return loss, lambda_ * penalty
        
        log.info("Starting retraining....")
        # Create MAS model that tracks small and big omega
        with self.strategy.scope():
            if self.mas_model is None:
                new_model = tf.keras.models.clone_model(model)
                new_model.set_weights(model.get_weights())
                self.mas_model = CustomModel(inputs=new_model.inputs, outputs=new_model.outputs)
                
                for weight in self.mas_model.weights:
                    self.mas_model.big_omega_var[weight.name] = tf.Variable(tf.zeros(weight.shape), trainable=False)
                    self.mas_model.small_omega_var[weight.name] = tf.Variable(tf.zeros(weight.shape), trainable=False)
                
            self.mas_model.compile(optimizer=optimizer, loss=mas_loss_fn)

            # Save previous weights for usage during training
            for weight in self.mas_model.trainable_weights:
                self.mas_model.previous_weights[weight.name] = tf.Variable(copy.deepcopy(weight), trainable=False)

        log.info("Fitting model....")
        self.mas_model.fit(train_set[0], train_set[1], epochs=epochs)
        
        # Calculate param importance matrix
        log.info("Calculating parameter importance....")
        if self.parameter_importance_via_gradient:
            self.create_importance_matrix_for_task_grad(dataset=train_set, frac=frac)
        else:
            self.create_importance_matrix_for_task_perturbation(dataset=train_set, delta=delta, frac=frac)

        if test_set is not None:
            log.info("Test-Set Loss: " + str(loss_fn(self.mas_model.predict(test_set[0]), test_set[1])))

        # return a copy of the model so that it can be used independently from the class
        return_model = tf.keras.models.clone_model(self.mas_model)
        return_model.set_weights(copy.deepcopy(self.mas_model.trainable_weights))
        
        return return_model
    
    
    def create_importance_matrix_for_task_perturbation(self, dataset, delta=0.0001, frac=1.):
        """
        Calculate the parameter importances.

        :param dataset: The dataset to base the parameter importance on. Could be a train or a test set.
        :param delta: Perturbation parameter to estimate parameter importance. 
        :param frac: The fraction of samples that should be used for importance matrix estimation. Use a small value for faster computation (0 < frac <= 1).
        """

        # Perturbate model weights and save old ones
        old_weights = copy.deepcopy(self.mas_model.weights)
        perturbed_weights = [weight + delta for weight in self.mas_model.trainable_weights]
        perturbed_model = CustomModel(inputs=self.mas_model.input, outputs=self.mas_model.output)
        perturbed_model.set_weights(perturbed_weights)

        x, y = dataset

        if isinstance(x, pd.DataFrame):
            x = x.to_numpy()
        if isinstance(y, pd.DataFrame):
            y = y.to_numpy()

        idxs = list(range(0, x.shape[0]))
        random.shuffle(idxs)
        num_samples = int(len(idxs)*frac)

        for idx in range(0, num_samples):
            with tf.GradientTape() as tape:
                y_pred = perturbed_model(np.expand_dims(x[idx], axis=0), training=True)  # Forward pass
                
                # Use L2-Norm over multiple outputs for better performance
                y_pred = tf.math.square(tf.math.l2_normalize(y_pred))
                y_true = tf.math.square(tf.math.l2_normalize(y[idx]))
                
                # Compute the loss value
                loss, _ = self.mas_model.loss(y_true=y_true, y_pred=y_pred)
            
            gradients = tape.gradient(loss, perturbed_model.trainable_variables)

            # Calculate small omega (accumulating gradients)
            for grad, var in zip(gradients, perturbed_model.trainable_variables):
                if grad is not None:
                    # Compute the update for small_omega_var
                    self.mas_model.small_omega_var[var.name].assign_add(tf.abs(grad/frac))
        
        small_omega_copy = copy.deepcopy(self.mas_model.small_omega_var)

        # Set model weights back to the original ones
        self.mas_model.set_weights(old_weights)

        # update small and big omega (parameter importance)
        for weight in self.mas_model.trainable_weights:
            self.mas_model.big_omega_var[weight.name].assign_add(tf.divide(small_omega_copy[weight.name], len(dataset[0])))
            self.mas_model.small_omega_var[weight.name].assign(self.mas_model.small_omega_var[weight.name] * 0.0)

    def create_importance_matrix_for_task_grad(self, dataset, frac=1.):
        """
        Calculate the parameter importances.

        :param dataset: The dataset to base the parameter importance on. Could be a train or a test set.
        :param frac: The fraction of samples that should be used for importance matrix estimation. Use a small value for faster computation (0 < frac <= 1).
        """
        x, y = dataset

        if isinstance(x, pd.DataFrame):
            x = x.to_numpy()
        if isinstance(y, pd.DataFrame):
            y = y.to_numpy()

        idxs = list(range(0, x.shape[0]))
        random.shuffle(idxs)
        num_samples = int(len(idxs)*frac)

        for idx in range(0, num_samples):
            with tf.GradientTape() as tape:
                y_pred = self.mas_model(np.expand_dims(x[idx], axis=0), training=True)  # Forward pass
                
                # Use L2-Norm over multiple outputs for better performance
                y_pred = tf.math.square(tf.math.l2_normalize(y_pred))
                y_true = tf.math.square(tf.math.l2_normalize(y[idx]))
                
                # Compute the loss value
                loss, _ = self.mas_model.loss(y_true=y_true, y_pred=y_pred)
        
            gradients = tape.gradient(loss, self.mas_model.trainable_variables)

            # Calculate small omega (accumulating gradients)
            for grad, var in zip(gradients, self.mas_model.trainable_variables):
                if grad is not None:
                    # Compute the update for small_omega_var
                    self.mas_model.small_omega_var[var.name].assign_add(tf.abs(grad/frac))

        # update small and big omega (parameter importance)
        for weight in self.mas_model.trainable_weights:
            self.mas_model.big_omega_var[weight.name].assign_add(tf.divide(self.mas_model.small_omega_var[weight.name], len(dataset[0])))
            self.mas_model.small_omega_var[weight.name].assign(self.mas_model.small_omega_var[weight.name] * 0.0)
