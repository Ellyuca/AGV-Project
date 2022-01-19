import tensorflow as tf
import tensorflow.python.keras.backend as K

class KerasModelMultiThread(object):

    @property 
    def layers(self):
        return self.model.layers

    def summary(self):
        self.model.summary()

    def add(self, layer):
        self.model.add(layer)

    def pop(self):
        self.model.pop()

    def predict(self, x):
        if self.mt_is_enable:
            with self.session.as_default():
                with self.graph.as_default():
                    return self.model.predict(x)
        else:
            return self.model.predict(x)

    def __init__(self, model, enable=True):
        self.model = model
        self.mt_is_enable = enable
        if enable:
            self.enable_multi_thread()

    def enable_multi_thread(self):
        #self.model._make_predict_function()
        self.session = K.get_session()
        self.graph = tf.get_default_graph()
        #self.graph.finalize()
        self.mt_is_enable = True
        pass
