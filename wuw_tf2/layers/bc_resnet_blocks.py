import tensorflow.compat.v2 as tf
from layers.sub_spectral_normalization import SubSpectralNormalization


@tf.keras.utils.register_keras_serializable()
class TransitionBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        filters=8,
        dilation=1,
        stride=1,
        padding="same",
        dropout=0.5,
        use_one_step=True,
        sub_groups=5,
        **kwargs
    ):
        super(TransitionBlock, self).__init__(**kwargs)
        self.filters = filters
        self.dilation = dilation
        self.stride = stride
        self.padding = padding
        self.dropout = dropout
        self.use_one_step = use_one_step
        self.sub_groups = sub_groups

        self.frequency_dw_conv = tf.keras.layers.DepthwiseConv2D(
            kernel_size=(1, 3),
            strides=(1, 1),
            dilation_rate=self.dilation,
            padding="same",
            use_bias=False,
        )
        if self.padding == "same":
            # print("hihihihi")
            self.temporal_dw_conv = tf.keras.layers.DepthwiseConv2D(
                kernel_size=(3, 1),
                strides=(1, 1),
                dilation_rate=self.dilation,
                padding="same",
                use_bias=False,
            )
        else:
            self.temporal_dw_conv = tf.keras.layers.DepthwiseConv2D(
                kernel_size=(3, 1),
                strides=(1, 1),
                dilation_rate=self.dilation,
                padding="valid",
                use_bias=False,
            )
        # print("stride: ", self.stride)
        self.batch_norm1 = tf.keras.layers.BatchNormalization()
        self.batch_norm2 = tf.keras.layers.BatchNormalization()
        self.conv1x1_1 = tf.keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=1,
            strides=self.stride,
            padding="valid",
            use_bias=False,
        )
        self.conv1x1_2 = tf.keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=1,
            strides=1,
            padding="valid",
            use_bias=False,
        )
        self.spatial_drop = tf.keras.layers.SpatialDropout2D(rate=self.dropout)
        self.spectral_norm = SubSpectralNormalization(self.sub_groups)

    def call(self, inputs):
        # expected input: [N, Time, Frequency, Channels]
        if inputs.shape.rank != 4:
            raise ValueError("input_shape.rank must be 4")

        net = inputs
        # print("inputs: ", net)
        net = self.conv1x1_1(net)
        # print("conv1x1_1: ", net)
        net = self.batch_norm1(net)
        net = tf.keras.activations.relu(net)
        net = self.frequency_dw_conv(net)
        # print("5: ", net)
        net = self.spectral_norm(net)

        residual = net
        # print("residual: ", residual)
        net = tf.keras.backend.mean(net, axis=2, keepdims=True)
        net = self.temporal_dw_conv(net)
        # print("6: ", net)
        net = self.batch_norm2(net)
        net = tf.keras.activations.swish(net)
        net = self.conv1x1_2(net)
        # print("conv1x1_2: ", net)
        net = self.spatial_drop(net)
        # print("8 :", net)
        net = net + residual
        # print("9: ", net)
        net = tf.keras.activations.relu(net)
        return net

    def get_config(self):
        config = {
            "filters": self.filters,
            "dilation": self.dilation,
            "stride": self.stride,
            "padding": self.padding,
            "dropout": self.dropout,
            "use_one_step": self.use_one_step,
            "sub_groups": self.sub_groups,
        }

        base_config = super(TransitionBlock, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def get_input_state(self):
        return self.temporal_dw_conv.get_input_state()

    def get_output_state(self):
        return self.temporal_dw_conv.get_output_state()


@tf.keras.utils.register_keras_serializable()
class NormalBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        filters,
        dilation=1,
        stride=1,
        padding="same",
        dropout=0.5,
        use_one_step=True,
        sub_groups=5,
        **kwargs
    ):
        super(NormalBlock, self).__init__(**kwargs)
        self.filters = filters
        self.dilation = dilation
        self.stride = stride
        self.padding = padding
        self.dropout = dropout
        self.use_one_step = use_one_step
        self.sub_groups = sub_groups

        self.frequency_dw_conv = tf.keras.layers.DepthwiseConv2D(
            kernel_size=(1, 3),
            strides=self.stride,
            dilation_rate=self.dilation,
            padding=self.padding,
            use_bias=False,
        )

        if self.padding == "same":
            self.temporal_dw_conv = tf.keras.layers.DepthwiseConv2D(
                kernel_size=(3, 1),
                strides=self.stride,
                dilation_rate=self.dilation,
                padding="same",
                use_bias=False,
            )
        else:
            self.temporal_dw_conv = tf.keras.layers.DepthwiseConv2D(
                kernel_size=(3, 1),
                strides=self.stride,
                dilation_rate=self.dilation,
                padding="valid",
                use_bias=False,
            )

        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.conv1x1 = tf.keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=1,
            strides=1,
            padding=self.padding,
            use_bias=False,
        )
        self.spatial_drop = tf.keras.layers.SpatialDropout2D(rate=self.dropout)
        self.spectral_norm = SubSpectralNormalization(self.sub_groups)

    def call(self, inputs):
        # expected input: [N, Time, Frequency, Channels]
        if inputs.shape.rank != 4:
            raise ValueError("input_shape.rank must be 4")

        identity = inputs
        net = inputs
        net = self.frequency_dw_conv(net)
        net = self.spectral_norm(net)

        residual = net
        net = tf.keras.backend.mean(net, axis=2, keepdims=True)
        net = self.temporal_dw_conv(net)
        net = self.batch_norm(net)
        net = tf.keras.activations.swish(net)
        net = self.conv1x1(net)
        net = self.spatial_drop(net)

        net = net + identity + residual
        net = tf.keras.activations.relu(net)

        return net

    def get_config(self):
        config = {
            "filters": self.filters,
            "dilation": self.dilation,
            "stride": self.stride,
            "padding": self.padding,
            "dropout": self.dropout,
            "use_one_step": self.use_one_step,
            "sub_groups": self.sub_groups,
        }
        base_config = super(NormalBlock, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def get_input_state(self):
        return self.temporal_dw_conv.get_input_state()

    def get_output_state(self):
        return self.temporal_dw_conv.get_output_state()
