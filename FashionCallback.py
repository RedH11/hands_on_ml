import tensorflow.keras as keras

class FashionCallback(keras.callbacks.Callback):
    # Set up to trigger on epoch end which is important to avoid catching a
    # volatile fluctuation and mistaking it for actual progress
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('loss') < 0.4:
            print("\nLoss is low, cancelling training")
            self.model.stop_training = True