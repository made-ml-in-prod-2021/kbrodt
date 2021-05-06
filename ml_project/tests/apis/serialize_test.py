from sklearn.pipeline import Pipeline

from clfit.apis import load_model, serialize_model


def test_serialize_model(tmp_path, trained_model, transformer):
    path = tmp_path / "model.pkl"
    serialize_model(trained_model, transformer, path)
    pipeline = load_model(path)

    assert isinstance(pipeline, Pipeline)

    model_loaded = pipeline.steps[-1][-1]
    assert trained_model.coef_.tolist() == model_loaded.coef_.tolist()
    assert trained_model.intercept_.tolist() == model_loaded.intercept_.tolist()
