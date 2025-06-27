# loss function and concept based on: https://www.sciencedirect.com/science/article/pii/S2405896320313823

import numpy as np
import tensorflow as tf
import copy
import random

class EWC:
    """
    Class for Elastic Weight Consolidation to tackle Concept Drift.
    This algorithm slows down learning on certain weights based on how important they are to previously seen tasks.
    """
    def __init__(self):
        self.prior_weights = []
        self.fisher_matrices = []
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

    def compute_fisher(self, model, data_samples, num_samples):
        """
        Compute the fisher matrix.

        :param model: The model that EWC shall be performed on.
        :param data_samples: Data samples of the training data of previous tasks (train_x).
        :param num_samples: The number of samples to be drawn from the data samples to calculate the fisher matrix.
        """
        weights = model.weights

        # Computation based on: https://github.com/stijani/elastic-weight-consolidation-tf2/blob/main/module.py
        # Initialize Fisher Information Matrix
        fisher_accum = np.array([np.zeros(layer.numpy().shape) for layer in weights], dtype=object)
        
        # Compute Fisher Information Matrix
        idxs = list(range(0, data_samples.shape[0]))
        random.shuffle(idxs)
        if num_samples > len(idxs):
            num_samples = len(idxs)
        for j in range(num_samples):
            with tf.GradientTape() as tape:
                pred = model(np.array([data_samples[idxs[j]]]))
            grads = tape.gradient(pred, weights)
            for m in range(len(weights)):
                fisher_accum[m] += np.square(grads[m])
        fisher_accum /= num_samples
        return fisher_accum
    
    def retrain(self, model, data_samples, epochs, loss_fn, optimizer, train_set, test_set=None, num_samples=30, lambda_=0.1):
        """
        Retrains the model to adapt to the new task presented in train_set.

        :param model: The model to be retrained.
        :param data_samples: Data samples of the training data of the prior model (train_x).
        :param epochs: The number of epochs to train for
        :param loss_fn: The loss function to use for training.
        :param optimizer: The optimizer to use for training.
        :param train_set: The Training Set holding data describing the concept drift.
        :param test_set: A Test Set that can be used during training for validation.
        :param num_samples: The number of samples to be drawn from the data samples to calculate the fisher matrix.
        :param lambda_: Parameter for tuning the EWC penalty. The lower lambda the more weight is given to the new task during retraining. 
        """
        print("Starting retraining....")
        self.prior_weights.append(copy.deepcopy(model.weights))
        print("Calculating parameter importance....")
        self.fisher_matrices.append(self.compute_fisher(model, data_samples, num_samples))
        
        # Define Loss function to be used during Retraining
        # loss function calculations are based on the formulars of: https://www.sciencedirect.com/science/article/pii/S2405896320313823
        def ewc_loss_fn(y_true, y_pred):
            loss = loss_fn(y_true, y_pred)
            penalty = 0.
            for f_mat, p_weights in zip(self.fisher_matrices, self.prior_weights):
                for f, v, w in zip(f_mat, model.weights, p_weights):
                    penalty += tf.math.reduce_sum(f * tf.math.square(v - w))
            return loss + 0.5 * lambda_ * penalty
        
        # Compile model to use new loss function but keep original weights
        with self.strategy.scope():
            new_model = tf.keras.models.clone_model(model)
            new_model.set_weights(model.get_weights())
            model = new_model
            model.compile(optimizer=optimizer, loss=ewc_loss_fn)
            model.set_weights(copy.deepcopy(self.prior_weights[-1]))
        print("Fitting model....")
        history = model.fit(train_set[0], train_set[1], epochs=epochs)
        
        if test_set is not None:
            print("Test-Set Loss: " + str(loss_fn(model.predict(test_set[0]), test_set[1])))

        return model
