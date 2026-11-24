"""Keep address-based legality identities alive until their history expires.

No identifiers or object graphs from this owner are observation features.
Deepcopy must rekey copied objects: retaining original integer keys could
otherwise let a later allocation overwrite a copied object's last owner.
"""
import copy


class IdentityLifetime:
    def __init__(self):
        self._objects = {}

    def remember(self, obj):
        identity = id(obj)
        self._objects[identity] = obj
        return identity

    def clear(self):
        self._objects.clear()

    def __len__(self):
        return len(self._objects)

    def __deepcopy__(self, memo):
        result = type(self)()
        memo[id(self)] = result
        for obj in self._objects.values():
            result.remember(copy.deepcopy(obj, memo))
        return result
