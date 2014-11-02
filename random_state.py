from sklearn.utils import check_random_state

class RandomStateSample(BaseEstimator, TransformerMixin):

    def __init__(self, random_state=None):
        self.random_state = random_state

    def fit(self, X=None, y=None, random_state=None):
        if random_state is None:
            random_state = self.random_state
        self._rng = check_random_state(random_state)