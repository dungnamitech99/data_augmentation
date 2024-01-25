import tensorflow.compat.v2 as tf


@tf.keras.utils.register_keras_serializable()
class SubSpectralNormalization(tf.keras.layers.Layer):
    def __init__(self, sub_groups, **kwargs):
        super(SubSpectralNormalization, self).__init__(**kwargs)
        self.sub_groups = sub_groups

        self.batch_norm = tf.keras.layers.BatchNormalization()

    def call(self, inputs):
        # expected input: [N, Time, Frequency, Channels]
        if inputs.shape.rank != 4:
            raise ValueError("input_shape.rank must be 4")

        # print("-----------------------------------------")
        # print("inputs_shape: ", inputs.shape)
        # print("inputs_shape_rank: ", inputs.shape.rank)
        # print("-----------------------------------------")
        input_shape = inputs.shape.as_list()
        if input_shape[2] % self.sub_groups:
            raise ValueError(
                f"input_shape[2]: {input_shape[2]} must be divisible by self.sub_groups: {self.sub_groups}"
            )

        net = inputs
        if self.sub_groups == 1:
            net = self.batch_norm(net)
        else:
            target_shape = [
                input_shape[1],
                input_shape[2] // self.sub_groups,
                input_shape[3] * self.sub_groups,
            ]

            net = tf.keras.layers.Reshape(target_shape)(net)
            net = self.batch_norm(net)
            net = tf.keras.layers.Reshape(input_shape[1:])(net)

        return net

    def get_config(self):
        config = {"sub_groups": self.sub_groups}
        base_config = super(SubSpectralNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
