import tensorflow as tf

from tensorflow.keras.applications import ConvNeXtTiny, DenseNet121
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    BatchNormalization,
    GlobalAveragePooling2D,
    GlobalMaxPooling2D,
    Concatenate,
    Multiply,
    Reshape,
    Conv2D,
    Add,
    Activation,
    Lambda,
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2


# ============================================================
# CBAM - Channel Attention
# ============================================================

def channel_attention(input_feature, ratio=8):
    channel = input_feature.shape[-1]

    shared_layer_one = Dense(
        channel // ratio,
        activation="relu",
        kernel_initializer="he_normal",
        use_bias=True,
    )

    shared_layer_two = Dense(
        channel,
        kernel_initializer="he_normal",
        use_bias=True,
    )

    # Average pooling
    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)

    # Max pooling
    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1, 1, channel))(max_pool)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)

    attention = Add()([avg_pool, max_pool])
    attention = Activation("sigmoid")(attention)

    return Multiply()([input_feature, attention])


# ============================================================
# CBAM - Spatial Attention
# ============================================================

def spatial_attention(input_feature):
    avg_pool = Lambda(
        lambda x: tf.reduce_mean(
            x,
            axis=3,
            keepdims=True
        )
    )(input_feature)

    max_pool = Lambda(
        lambda x: tf.reduce_max(
            x,
            axis=3,
            keepdims=True
        )
    )(input_feature)

    concat = Concatenate(axis=3)([
        avg_pool,
        max_pool
    ])

    attention = Conv2D(
        filters=1,
        kernel_size=7,
        padding="same",
        activation="sigmoid",
        kernel_initializer="he_normal",
        use_bias=False,
    )(concat)

    return Multiply()([
        input_feature,
        attention
    ])


# ============================================================
# CBAM
# ============================================================

def CBAM(input_feature, ratio=8):
    x = channel_attention(
        input_feature,
        ratio
    )

    x = spatial_attention(x)

    return x


# ============================================================
# Build Hybrid ConvNeXt + DenseNet121 + CBAM Model
# ============================================================

def build_model():
    """
    Reconstructs the exact architecture used during Kaggle
    training.

    Architecture:
        Input
          |
        +---------------------+
        |                     |
    ConvNeXtTiny         DenseNet121
        |                     |
       CBAM                  CBAM
        |                     |
       GAP                   GAP
        |                     |
        +------ Concatenate --+
                  |
             BatchNorm
                  |
              Dense 512
                  |
             BatchNorm
                  |
              Dropout
                  |
              Dense 256
                  |
             BatchNorm
                  |
              Dropout
                  |
             Softmax (4)
    """

    tf.keras.backend.clear_session()

    # --------------------------------------------------------
    # Input
    # --------------------------------------------------------

    input_layer = Input(
        shape=(224, 224, 3),
        name="input_image"
    )

    # --------------------------------------------------------
    # ConvNeXtTiny
    # --------------------------------------------------------

    convnext = ConvNeXtTiny(
        weights=None,
        include_top=False,
        input_tensor=input_layer
    )

    # --------------------------------------------------------
    # DenseNet121
    # --------------------------------------------------------

    densenet = DenseNet121(
        weights=None,
        include_top=False,
        input_tensor=input_layer
    )

    # --------------------------------------------------------
    # Stage-2 trainability configuration
    #
    # This matches the model from which the deployment
    # weights were created.
    # --------------------------------------------------------

    convnext.trainable = False
    densenet.trainable = False

    for layer in convnext.layers[-60:]:
        layer.trainable = True

    for layer in densenet.layers[-60:]:
        layer.trainable = True

    # --------------------------------------------------------
    # ConvNeXt feature branch + CBAM
    # --------------------------------------------------------

    conv_features = convnext(input_layer)

    conv_features = CBAM(
        conv_features,
        ratio=8
    )

    conv_features = GlobalAveragePooling2D(
        name="convnext_gap"
    )(conv_features)

    # --------------------------------------------------------
    # DenseNet feature branch + CBAM
    # --------------------------------------------------------

    dense_features = densenet.output

    dense_features = CBAM(
        dense_features,
        ratio=8
    )

    dense_features = GlobalAveragePooling2D(
        name="densenet_gap"
    )(dense_features)

    # --------------------------------------------------------
    # Feature Fusion
    # --------------------------------------------------------

    fusion = Concatenate(
        name="feature_fusion"
    )([
        conv_features,
        dense_features
    ])

    fusion = BatchNormalization(
        name="fusion_bn"
    )(fusion)

    # --------------------------------------------------------
    # Classification Head
    # --------------------------------------------------------

    x = Dense(
        512,
        activation="relu",
        kernel_initializer="he_normal",
        kernel_regularizer=l2(1e-4),
        name="fc1"
    )(fusion)

    x = BatchNormalization(
        name="bn1"
    )(x)

    x = Dropout(
        0.5,
        name="dropout1"
    )(x)

    x = Dense(
        256,
        activation="relu",
        kernel_initializer="he_normal",
        kernel_regularizer=l2(1e-4),
        name="fc2"
    )(x)

    x = BatchNormalization(
        name="bn2"
    )(x)

    x = Dropout(
        0.3,
        name="dropout2"
    )(x)

    output = Dense(
        4,
        activation="softmax",
        name="predictions"
    )(x)

    # --------------------------------------------------------
    # Final Model
    # --------------------------------------------------------

    model = Model(
        inputs=input_layer,
        outputs=output,
        name="Hybrid_ConvNeXt_DenseNet121_CBAM"
    )

    return model