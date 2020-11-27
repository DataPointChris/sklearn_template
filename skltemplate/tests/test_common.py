import pytest

from sklearn.utils.estimator_checks import check_estimator

from sklearn_template import TemplateEstimator
from sklearn_template import TemplateClassifier
from sklearn_template import TemplateTransformer


@pytest.mark.parametrize(
    "Estimator", [TemplateEstimator, TemplateTransformer, TemplateClassifier]
)
def test_all_estimators(Estimator):
    return check_estimator(Estimator)
