import torch


def accuracy(y_hat, y, num_classes, mask=None):
    """Compute the number of correct predictions.

    Defined in :numref:`sec_softmax_scratch`"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = torch.argmax(y_hat, axis=1)
    elif num_classes == 2:
        y_hat = y_hat > 0.5
    cmp = y_hat.to(dtype=y.dtype) == y
    cmp = cmp.to(dtype=y.dtype)[mask.bool()]
    return float(cmp.sum()) / mask.sum()


def true_positive(pred, target, num_classes, mask=None):
    r"""Computes the number of true positive predictions.

    Args:
        pred (Tensor): The predictions.
        target (Tensor): The targets.
        num_classes (int): The number of classes.

    :rtype: :class:`LongTensor`
    """
    if len(pred.shape) > 1 and pred.shape[1] > 1:
        pred = torch.argmax(pred, axis=1)
    elif num_classes == 2:
        pred = pred > 0.5
    out = []
    if mask is None:
        mask = torch.ones(pred.shape)
    for i in range(num_classes):
        out.append((((pred == i) & (target == i)) & mask).sum())

    return torch.tensor(out)


def true_negative(pred, target, num_classes, mask=None):
    r"""Computes the number of true negative predictions.

    Args:
        pred (Tensor): The predictions.
        target (Tensor): The targets.
        num_classes (int): The number of classes.

    :rtype: :class:`LongTensor`
    """
    if len(pred.shape) > 1 and pred.shape[1] > 1:
        pred = torch.argmax(pred, axis=1)
    elif num_classes == 2:
        pred = pred > 0.5
    out = []
    if mask is None:
        mask = torch.ones(pred.shape)
    for i in range(num_classes):
        out.append((((pred != i) & (target != i)) & mask).sum())

    return torch.tensor(out)


def false_positive(pred, target, num_classes, mask=None):
    r"""Computes the number of false positive predictions.

    Args:
        pred (Tensor): The predictions.
        target (Tensor): The targets.
        num_classes (int): The number of classes.

    :rtype: :class:`LongTensor`
    """
    if len(pred.shape) > 1 and pred.shape[1] > 1:
        pred = torch.argmax(pred, axis=1)
    elif num_classes == 2:
        pred = pred > 0.5
    out = []
    if mask is None:
        mask = torch.ones(pred.shape)
    for i in range(num_classes):
        out.append((((pred == i) & (target != i)) & mask).sum())

    return torch.tensor(out)


def false_negative(pred, target, num_classes, mask=None):
    r"""Computes the number of false negative predictions.

    Args:
        pred (Tensor): The predictions.
        target (Tensor): The targets.
        num_classes (int): The number of classes.

    :rtype: :class:`LongTensor`
    """
    if len(pred.shape) > 1 and pred.shape[1] > 1:
        pred = torch.argmax(pred, axis=1)
    elif num_classes == 2:
        pred = pred > 0.5
    out = []
    if mask is None:
        mask = torch.ones(pred.shape)
    for i in range(num_classes):
        out.append((((pred != i) & (target == i)) & mask).sum())

    return torch.tensor(out)


def precision(pred, target, num_classes, mask=None):
    r"""Computes the precision:
    :math:`\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FP}}`.

    Args:
        pred (Tensor): The predictions.
        target (Tensor): The targets.
        num_classes (int): The number of classes.

    :rtype: :class:`Tensor`
    """
    if len(pred.shape) > 1 and pred.shape[1] > 1:
        pred = torch.argmax(pred, axis=1)
    elif num_classes == 2:
        pred = pred > 0.5
    if mask is None:
        mask = torch.ones(pred.shape)
    tp = true_positive(pred, target, num_classes, mask).to(torch.float)
    fp = false_positive(pred, target, num_classes, mask).to(torch.float)

    out = tp / (tp + fp)
    out[torch.isnan(out)] = 0

    return out


def recall(pred, target, num_classes, mask=None):
    r"""Computes the recall:
    :math:`\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}}`.

    Args:
        pred (Tensor): The predictions.
        target (Tensor): The targets.
        num_classes (int): The number of classes.

    :rtype: :class:`Tensor`
    """
    if len(pred.shape) > 1 and pred.shape[1] > 1:
        pred = torch.argmax(pred, axis=1)
    elif num_classes == 2:
        pred = pred > 0.5
    if mask is None:
        mask = torch.ones(pred.shape)
    tp = true_positive(pred, target, num_classes, mask).to(torch.float)
    fn = false_negative(pred, target, num_classes, mask).to(torch.float)

    out = tp / (tp + fn)
    out[torch.isnan(out)] = 0

    return out


def f_beta_score(pred, target, num_classes, mask=None, beta=1):
    r"""Computes the :math:`F_beta` score:
    :math:`2 \cdot \frac{\mathrm{precision} \cdot \mathrm{recall}}
    {\mathrm{precision}+\mathrm{recall}}`.

    Args:
        pred (Tensor): The predictions.
        target (Tensor): The targets.
        num_classes (int): The number of classes.

    :rtype: :class:`Tensor`
    """
    if len(pred.shape) > 1 and pred.shape[1] > 1:
        pred = torch.argmax(pred, axis=1)
    elif num_classes == 2:
        pred = pred > 0.5
    if mask is None:
        mask = torch.ones(pred.shape)
    prec = precision(pred, target, num_classes, mask)
    rec = recall(pred, target, num_classes, mask)

    score = (1 + beta ** 2) / (beta ** 2 / rec + 1 / prec)
    score[torch.isnan(score)] = 0

    return score
