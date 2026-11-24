"""Declared fixed-batch FP32 policy numerics for paired terminal evidence."""
from contextlib import nullcontext
from .interactive_evaluation import InteractiveSelfPlayEvaluator

class FixedBatchInteractiveEvaluator(InteractiveSelfPlayEvaluator):
    def _autocast(self):return nullcontext()
    def _choose_many(self,requests):
        results=[]
        for start in range(0,len(requests),self.batch_size):
            chunk=requests[start:start+self.batch_size]
            padded=chunk+[chunk[-1]]*(self.batch_size-len(chunk))
            results.extend(super()._choose_many(padded)[:len(chunk)])
        return results
