class RangeLookup:
    def __init__(self, lookup, lower_bound_inclusive=True):
        assert isinstance(lookup, dict), "lookup must be a dictionary"
        assert all(
            isinstance(k, tuple) and len(k) == 2 for k in lookup.keys()
        ), "all keys in lookup must be tuples that contain 2 elements"

        self.lookup = lookup
        self.lower_bound_inclusive = lower_bound_inclusive

    def _getitem_lower_bound_inclusive(self, idx):
        for (_min, _max), value in self.lookup.items():
            if idx >= _min and idx < _max:
                return value

        raise IndexError()

    def _getitem_lower_bound_exclusive(self, idx):
        for (_min, _max), value in self.lookup.items():
            if idx > _min and idx <= _max:
                return value

        raise IndexError()

    def __getitem__(self, idx):
        if self.lower_bound_inclusive:
            return self._getitem_lower_bound_inclusive(idx)
        else:
            return self._getitem_lower_bound_exclusive(idx)
