import pandas as pd
import numpy as np
import warnings

class TripleBarrierLabel:
    """
    Triple-Barrier labeling for financial time series.

    Parameters
    ----------
    data : pandas.Series or pandas.DataFrame
        Price series (usually close prices). If a DataFrame with one column is provided,
        it will be squeezed to a Series.
    horizon : int
        Maximum look-ahead window (number of steps forward) for applying the barriers.
        Must be strictly positive.

    Attributes
    ----------
    labels : np.ndarray of shape (n_samples,)
        The class labels: +1 (upper barrier hit first), -1 (lower barrier hit first), or 0 (no barrier hit).
    time_to_hit : np.ndarray of shape (n_samples,)
        Number of steps forward (within the horizon) until a barrier was hit.
        Equals `horizon` if no barrier was hit.
    barriers : tuple of (np.ndarray, np.ndarray)
        Tuple (lower_barriers, upper_barriers) containing the dynamic barrier values for each observation.
    volatility_function : np.ndarray of shape (n_samples,)
        Volatility estimates used to compute the barriers (rolling std or exponential std).
    windows : list of np.ndarray
        Forward price windows per observation, each of length = horizon (prices [t+1, …, t+horizon]).
    """


    def __init__(self, data, horizon):
        if not isinstance(data, (pd.DataFrame, pd.Series)):
            raise ValueError("'data' must be pandas DataFrame or pandas Series")
        self._data = data

        if horizon <= 0:
            raise ValueError('horizon must be > 0')
        if not isinstance(horizon, int):
            raise ValueError('horizon must be int')
        self._horizon = horizon

    def fit_labels(self, up_t, low_t, volatility_func='moving_std', volatility_window=None, ewm_params=None):
        """
        Fit the triple-barrier labels.

        Parameters
        ----------
        up_t : float
            Multiplier for the upper barrier (in units of volatility).
        low_t : float
            Multiplier for the lower barrier (in units of volatility).
        volatility_func : {'moving_std', 'exponential'}, default='moving_std'
            Volatility estimation method:
            - 'moving_std' : rolling standard deviation with user-specified window (`volatility_window`).
            - 'exponential' : exponentially weighted std, requires `ewm_params`.
        volatility_window : int, optional
            Window size used when `volatility_func='moving_std'`.
        ewm_params : dict, optional
            Parameters passed to pandas.Series.ewm() when using exponential volatility.
            Ignored if `volatility_func != 'exponential'`.

        Returns
        -------
        self : TripleBarrierLabel
            The fitted instance (for method chaining).
        """

        if volatility_func not in ('moving_std', 'exponential'):
            raise ValueError("'volatility_func' must be one of: 'moving_std', 'exponential'")

        if ewm_params and volatility_func != 'exponential':
            warnings.warn(f"volatility function is {volatility_func}. 'ewm_params' are not in use")

        
        if volatility_func == 'moving_std':
            if volatility_window is None:
                raise ValueError("volatility function is 'moving_std'; 'volatility_window' must be provided")
            if not isinstance(volatility_window, int):
                raise ValueError("'volatility_window' must be an integer")
            self._vol_func = self._data.rolling(volatility_window).std().shift(1).to_numpy().squeeze()

        elif volatility_func == 'exponential':
            if ewm_params is None:
                raise ValueError("'ewm_params' must be provided when using exponential volatility")
            if not isinstance(ewm_params, dict):
                raise ValueError("'ewm_params' must be a dict")
            if volatility_window is not None:
                warnings.warn("volatility function is 'exponential'; 'volatility_window' is not in use")
            self._vol_func = self._data.ewm(**ewm_params).std().shift(1).to_numpy().squeeze()
            

        def _make_windows(data, horizon):
            arr = data.values
            p, w = [], []
            for i in range(len(arr) - horizon + 1):
                p.append(arr[i])
                w.append(arr[i+1 : i+horizon+1])
            return p, w
        self._p, self._w = _make_windows(self._data, self._horizon)
        

        labels, time_to_hit, up_barriers, low_barriers = [], [], [], []
        for i, (p, w) in enumerate(zip(self._p, self._w)):
            # Compute dynamic upper/lower barriers around price p based on local volatility
            ub = p * (1 + up_t * self._vol_func[i])
            lb = p * (1 - low_t * self._vol_func[i])
            up_barriers.append(ub)
            low_barriers.append(lb)
        
            # Find the first step in the window where each barrier is hit (if any)
            hit_up = (w >= ub).argmax() if (w >= ub).any() else None
            hit_low = (w <= lb).argmax() if (w <= lb).any() else None
        
            # Assign label based on which barrier is hit first
            if hit_up is not None and (hit_low is None or hit_up <= hit_low):
                labels.append(1)
                time_to_hit.append(hit_up + 1)   # +1 because argmax is index within window
            elif hit_low is not None:
                labels.append(-1)
                time_to_hit.append(hit_low + 1)
            else:
                labels.append(0)
                time_to_hit.append(self._horizon)  # No barrier hit; full horizon

        
        self._labels = np.array(labels, dtype=int)
        self._time_to_hit = np.array(time_to_hit, dtype=int)
        self._ub = np.array(up_barriers, dtype=float)
        self._lb = np.array(low_barriers, dtype=float)
        
        return self

    @property
    def labels(self):
        """
        np.ndarray of shape (n_samples,)
            Triple-barrier class labels for each observation:
            +1 if the upper barrier was hit first,
            -1 if the lower barrier was hit first,
             0 if no barrier was hit within the horizon.
        """
        if not hasattr(self, '_labels'):
            raise AttributeError('no labels were found, run fit_labels() first')
        return self._labels
    
    @property
    def time_to_hit(self):
        """
        np.ndarray of shape (n_samples,)
            Number of steps forward until the first barrier was hit.
            Equals `horizon` if no barrier was hit within the horizon.
        """
        if not hasattr(self, '_time_to_hit'):
            raise AttributeError('no times were found, run fit_labels() first')
        return self._time_to_hit

    @property
    def average_uniqueness_score(self):
        """
        np.ndarray of shape (n_samples,)
            Average uniqueness score for each observation.
            Computed as the mean inverse concurrency across the
            label’s active window. Values range from 0 to 1, where
            higher values indicate lower overlap with other labels.
        """
        if not hasattr(self, '_labels'):
            raise AttributeError('no labels were found, run fit_labels() first')
    
        t1 = np.zeros(shape=len(self._data))
        for i, t in enumerate(self._time_to_hit):
            t1[i: i+t] += 1
    
        weights = []
        for i, t in enumerate(self._time_to_hit):
            duration = t1[i: i+t]
            weights.append(np.mean(1 / duration))
    
        return np.array(weights)

    def sample_weights(self, alpha=0.01, clip_min=0.1):
        """
        Compute final sample weights for model training.

        Parameters
        ----------
        alpha : float, default=0.01
            Decay rate used in the exponential time-decay term.
            Higher values increase the penalty on events with
            longer time_to_hit, reducing their influence.
        clip_min : float, default=0.1
            Minimum allowed weight after normalization. Values
            below this threshold are clipped to `clip_min` to
            prevent observations from having negligible impact.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Final sample weights combining:
            - average uniqueness score, which down-weights highly
              overlapping labels, and
            - exponential time-decay based on time_to_hit.
            The weights are normalized by their mean and then
            clipped from below at `clip_min`.
        """
        uniq = self.average_uniqueness_score
        decay = np.exp(-alpha * self._time_to_hit)
        w = uniq * decay
        return np.clip(w / w.mean(), clip_min, 1.0)
    
    @property
    def barriers(self):
        """
        tuple of (np.ndarray, np.ndarray)
            Dynamic barrier values for each observation:
            (lower_barriers, upper_barriers).
            Each array has shape (n_samples,).
        """
        if not hasattr(self, '_ub'):
            raise AttributeError('no barriers were found, run fit_labels() first')
        return self._lb, self._ub
    
    @property
    def volatility_function(self):
        """
        np.ndarray of shape (n_samples,)
            Volatility estimates used to scale the dynamic barriers.
            Computed using either rolling standard deviation or
            exponentially weighted standard deviation, depending
            on the chosen method in `fit_labels`.
        """
        if not hasattr(self, '_vol_func'):
            raise AttributeError('no volatility function found, run fit_labels() first')
        return self._vol_func
    
    @property
    def windows(self):
        """
        list of np.ndarray
            Forward price windows for each observation.
            Each window has length = horizon and contains
            prices [P_{t+1}, ..., P_{t+horizon}].
        """
        if not hasattr(self, '_w'):
            raise AttributeError('no windows were found, run fit_labels() first')
        return self._w