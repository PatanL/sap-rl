"""Shared total-work cap across tree, screening and confirmation clones."""
import time
from dataclasses import dataclass,field

class SearchBudgetExceeded(RuntimeError):pass

@dataclass
class SearchWorkBudget:
    max_decisions:int=20000
    max_neural_states:int=20000
    seconds:float=30.
    clock:object=time.monotonic
    decisions:int=0
    neural_states:int=0
    forecast_contexts:int=0
    started:float=field(init=False)
    def __post_init__(self):
        if min(self.max_decisions,self.max_neural_states,self.seconds)<=0:raise ValueError('Positive search budgets required')
        self.started=self.clock()
    def check(self):
        if self.clock()-self.started>=self.seconds:raise SearchBudgetExceeded('deadline')
    def step(self):
        self.check()
        if self.decisions>=self.max_decisions:raise SearchBudgetExceeded('total decision transitions')
        self.decisions+=1
    def neural(self,count=1):
        self.check()
        if self.neural_states+count>self.max_neural_states:raise SearchBudgetExceeded('neural state evaluations')
        self.neural_states+=count
    def report(self):return {'decisions':self.decisions,'neuralStates':self.neural_states,'forecastContexts':self.forecast_contexts,
        'wallSeconds':self.clock()-self.started,'maxDecisions':self.max_decisions,'maxNeuralStates':self.max_neural_states,
        'deadlineSeconds':self.seconds,'cooperativeDeadline':True}
