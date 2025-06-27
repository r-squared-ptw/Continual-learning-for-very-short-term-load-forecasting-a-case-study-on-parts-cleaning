import tensorflow as tf
import copy
from tensorflow.keras import Model

# CustomModel class to override parts of the tf training loop (loss function needs x values of current training set in a batchwise manner).
class CustomModel(Model):
    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            loss = self.loss(y_true=y, y_pred=y_pred, x=x)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
    

class LWF:
    """
    Class for Learning without Forgetting to tackle Concept Drift.
    The algorithm introduces a penalty during training to make sure the updated models prediction won't differ too much from the original ones.
    """
    def __init__(self):
        self.prior_models = []
        self.prior_weights = []
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

    def retrain(self, model, epochs, loss_fn, optimizer, train_set, test_set=None, penalty_loss_fn=tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE), lambda_=0.1):
        """
        Retrains the model to adapt to the new task presented in train_set.

        :param model: The model to be retrained.
        :param epochs: The number of epochs to train for
        :param loss_fn: The loss function to use for training.
        :param optimizer: The optimizer to use for training.
        :param train_set: The Training Set holding data describing the concept drift.
        :param test_set: A Test Set that can be used during training for validation.
        :param penalty_loss_fn: The loss function that is used to compute the penalty term. 
        :param _lambda: Regularization parameter. The higher the more the model is penalized for straying too far from the original predictions.
        """
        print("Starting retraining....")
        
        # save models of prior tasks for penalty calculations
        self.prior_models.append(tf.keras.models.clone_model(model))
        self.prior_models[-1].set_weights(copy.deepcopy(model.weights))

        # loss function calculations are based on the formulars of: https://www.sciencedirect.com/science/article/pii/S2405896320313823
        def lwf_loss_fn(y_true, y_pred, x):
            loss = loss_fn(y_true, y_pred)
            penalty = 0.0
            # Compute penalty by iterating over models of previous tasks and computing their output on the current training batch
            for old_model in self.prior_models:
                y_pred_old = old_model(x, training=False)
                penalty += penalty_loss_fn(y_pred, y_pred_old)

            return (1 - lambda_) * loss + (lambda_/len(self.prior_models)) * penalty
        
        # Compile model to use new loss function but keep original weights
        with self.strategy.scope():
            new_model = tf.keras.models.clone_model(model)
            new_model.set_weights(model.get_weights())
            model2 = CustomModel(inputs=new_model.inputs, outputs=new_model.outputs)
            model2.build(input_shape=model.layers[0].input_shape)
            model2.compile(optimizer=optimizer, loss=lwf_loss_fn)
            model2.set_weights(copy.deepcopy(self.prior_models[-1].weights))
        print("Fitting model....")
        model2.fit(train_set[0], train_set[1], epochs=epochs)

        if test_set is not None:
            print("Test-Set Loss: " + str(loss_fn(model2.predict(test_set[0]), test_set[1])))

        return model2