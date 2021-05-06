from sklearn.utils.validation import check_is_fitted

from clfit.apis import train_model


def test_train_model(model, features, target):
    train_model(model, features, target)

    check_is_fitted(model)
