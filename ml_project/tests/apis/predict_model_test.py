from clfit.apis import predict_model


def test_train_model(trained_model, features, target):
    predicts = predict_model(trained_model, features, return_proba=False)

    assert predicts.shape == target.shape

    predicts = predict_model(trained_model, features, return_proba=True)
    assert predicts.shape == target.shape
