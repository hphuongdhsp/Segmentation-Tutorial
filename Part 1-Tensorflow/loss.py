import tensorflow.keras.backend as K
from tensorflow.keras.losses import binary_crossentropy


def binary_accuracy(y_true, y_pred):
    proba = 0.5
    return K.mean(K.equal(K.round(y_true), K.round(y_pred / (2 * proba))), axis=-1)


def jaccard(y_true, y_pred):
    smooth = 0.001
    y_true_f = K.batch_flatten(K.round(y_true))
    y_pred_f = K.batch_flatten(K.round(y_pred))
    intersection = K.sum(y_true_f * y_pred_f, axis=-1)
    jac = (smooth + intersection) / (
        smooth - intersection + K.sum(y_pred_f, axis=-1) + K.sum(y_true_f, axis=-1)
    )
    return jac


def t_scoreb(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    y1 = y_pred_f * y_true_f
    y2 = (1 - y_pred_f) * (1 - y_true_f)
    k = 2.5
    f1 = 1 - (1 - y1) ** k
    f2 = (1 - y2) ** k
    score = K.sum(f1) / K.sum(f2)
    return score


def t_lossb(y_true, y_pred):
    return 1 - t_scoreb(y_true, y_pred)


def dice_coeffi(y_true, y_pred):
    smooth = 10 ** -6
    # Workaround for shape bug. For some reason y_true shape was not being set correctly
    #    y_true.set_shape(y_pred.get_shape())
    # Without K.clip, K.sum() behaves differently when compared to np.count_nonzero()
    y_true_f = K.batch_flatten(y_true)
    y_pred_f = K.batch_flatten(y_pred)
    intersection = 2 * K.sum(y_true_f * y_pred_f, axis=1)
    #    union = K.sum(y_true_f * y_true_f, axis=1) + K.sum(y_pred_f * y_pred_f, axis=1)
    union = K.sum(y_true_f, axis=1) + K.sum(y_pred_f, axis=1)
    return K.mean((smooth + intersection) / (smooth + union))


def dice_lossi(y_true, y_pred):
    loss = 1 - dice_coeffi(y_true, y_pred)
    return loss


def bce_dice_lossi(y_true, y_pred):
    # Workaround for shape bug.
    #    y_true.set_shape(y_pred.get_shape())

    # Without K.clip, K.sum() behaves differently when compared to np.count_nonzero()
    y_true_f = K.clip(K.batch_flatten(y_true), K.epsilon(), 1.0)
    y_pred_f = K.clip(K.batch_flatten(y_pred), K.epsilon(), 1.0)

    bce = K.mean(binary_crossentropy(y_true_f, y_pred_f))
    dice = dice_lossi(y_true, y_pred)
    return bce + dice


def dice_coeffb(y_true, y_pred):
    smooth = 10 ** -6
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = 2 * K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f)
    return (smooth + intersection) / (smooth + union)


def dice_lossb(y_true, y_pred):
    loss = 1 - dice_coeffb(y_true, y_pred)
    return loss


def bce_dice_lossb(y_true, y_pred):
    y_true_f = K.clip(K.batch_flatten(y_true), K.epsilon(), 1.0)
    y_pred_f = K.clip(K.batch_flatten(y_pred), K.epsilon(), 1.0)
    bce = binary_crossentropy(y_true_f, y_pred_f)
    dice = dice_lossb(y_true, y_pred)
    return bce + dice
