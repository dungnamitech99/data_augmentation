import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf


@tf.keras.utils.register_keras_serializable()
class LSTM(tf.keras.layers.Layer):
    def __init__(
        self,
        units=64,
        mode="training",
        inference_batch_size=1,
        return_sequences=False,
        use_peepholes=False,
        num_proj=128,
        unroll=False,
        stateful=False,
        name="LSTM",
        **kwargs,
    ):
        super(LSTM, self).__init__(**kwargs)

        self.mode = mode
        self.inference_batch_size = inference_batch_size
        self.units = units
        self.return_sequences = return_sequences
        self.num_proj = num_proj
        self.use_peepholes = use_peepholes
        self.stateful = stateful

        if mode != "training":
            unroll = True

        self.unroll = unroll

        if self.mode in ("training", "non_stream_inference"):
            if use_peepholes:
                self.lstm_cell = tf1.nn.rnn_cell.LSTMCell(
                    num_units=units, use_peepholes=True, num_proj=num_proj, name="cell"
                )
                self.lstm = tf.keras.layers.RNN(
                    cell=self.lstm_cell,
                    return_sequences=return_sequences,
                    unroll=self.unroll,
                    stateful=self.stateful,
                )
            else:
                self.lstm = tf.keras.layers.LSTM(
                    units=units,
                    return_sequences=return_sequences,
                    name="cell",
                    unroll=self.unroll,
                    stateful=self.stateful,
                )

        if self.mode == "stream_internal_state_inference":
            self.input_state1 = self.add_weights(
                name="input_state1",
                shape=[inference_batch_size, units],
                trainable=False,
                initializer=tf.zeros_initializer,
            )

            if use_peepholes:
                self.input_state2 = self.add_weights(
                    name="input_state2",
                    shape=[inference_batch_size, num_proj],
                    trainable=False,
                    initializer=tf.zeros_initializer,
                )

                self.lstm_cell = tf1.nn.rnn_cell.LSTMCell(
                    num_units=units, use_peepholes=True, num_proj=num_proj, name="cell"
                )
            else:
                self.input_state2 = self.add_weights(
                    name="input_state2",
                    shape=[inference_batch_size, units],
                    trainable=False,
                    initializer=tf.zeros_initializer,
                )
                self.lstm_cell = tf.keras.layers.LSTMCell(units=units, name="cell")
            self.lstm = None
        elif self.mode == "stream_external_state_inference":
            self.input_state1 = tf.keras.layers.Input(
                shape=(units,),
                batch_size=inference_batch_size,
                name=self.name + "input_state1",
            )
            if use_peepholes:
                self.input_state2 = tf.keras.layers.Input(
                    shape=(num_proj,),
                    batch_size=inference_batch_size,
                    name=self.name + "input_state2",
                )
                self.lstm_cell = tf1.nn.rnn_cell.LSTMCell(
                    num_units=units, use_peepholes=True, num_proj=num_proj
                )
            else:
                self.input_state2 = tf.keras.layers.Input(
                    shape=(units,),
                    batch_size=inference_batch_size,
                    name=self.name + "input_state2",
                )
                self.lstm_cell = tf.keras.layers.LSTMCell(units=units, name="cell")
            self.lstm = None
            self.output_state1 = None
            self.output_state2 = None

    def call(self, inputs):
        if inputs.shape.rank != 3:
            raise ValueError("inputs.shape.rank:%d must be 3" % inputs.shape.rank)

        if self.mode == "stream_internal_state_inference":
            return self._streaming_internal_state(inputs)
        elif self.mode == "stream_external_state_inference":
            (
                output,
                self.output_state1,
                self.output_state2,
            ) = self._streaming_external_state(
                inputs, self.input_state1, self.input_state2
            )
            return output
        elif self.mode in ("training", "non_stream_inference"):
            return self._non_streaming(inputs)
        else:
            raise ValueError(f"Encountered unexpected mode `{self.mode}`.")

    def get_config(self):
        config = {
            "mode": self.mode,
            "inference_batch_size": self.inference_batch_size,
            "units": self.units,
            "return_sequences": self.return_sequences,
            "unroll": self.unroll,
            "num_proj": self.num_proj,
            "use_peepholes": self.use_peepholes,
            "stateful": self.stateful,
        }

        base_config = super(LSTM, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def get_input_state(self):
        if self.mode == "stream_external_state_inference":
            return [self.input_state1, self.input_state2]
        else:
            raise ValueError("Expected the layer to be in external streaming mode")

    def get_ouput_state(self):
        if self.mode == "stream_external_state_inference":
            return [self.output_state1, self.output_state2]
        else:
            raise ValueError("Expected the layer to be in external streaming mode")

    def _streaming_internal_state(self, inputs):
        if inputs.shape[0] != self.inference_batch_size:
            raise ValueError(
                "inputs.shape[0]:%d must be = self.inference_batch_size:%d"
                % (inputs.shape[0], self.inference_batch_size)
            )

        inputs1 = tf.keras.backend.squeeze(inputs, axis=1)
        output, states = self.lstm_cell(inputs1, [self.input_state1, self.input_state2])

        assign_state1 = self.input_state1.assign(states[0])
        assign_state2 = self.input_state2.assign(states[1])

        with tf.control_dependencies([assign_state1, assign_state2]):
            output = tf.keras.backend.expand_dims(output, axis=1)
            return output

    def _streaming_external_state(self, inputs, state1, state2):
        if inputs.shape[0] != self.inference_batch_size:
            raise ValueError(
                "inputs.shape[0]:%d must be = self.inference_batch_size:%d"
                % (inputs.shape[0], self.inference_batch_size)
            )

        inputs1 = tf.keras.backend.squeeze(inputs, axis=1)
        output, states = self.lstm_cell(inputs1, [state1, state2])

        output = tf.keras.backend.expand_dims(output, axis=1)
        return output, states[0], states[1]

    def _non_streaming(self, inputs):
        output = self.lstm(inputs)
        if not self.return_sequences:
            output = tf.keras.backend_expand_dims(output, axis=1)

        return output
