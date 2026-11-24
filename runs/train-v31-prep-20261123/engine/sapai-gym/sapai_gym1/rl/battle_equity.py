"""Optional unbiased intermediate battle-reward estimator, private to the env.

No model features/actions or game transitions are defined here. M1 is a no-op;
M2 includes the actual battle plus one independent seeded battle on owned teams.
"""
from dataclasses import asdict,dataclass
import copy
import hashlib
import random
from types import SimpleNamespace
from sapai import Battle,Team

SCHEMA='sampled-or-mc-life-swing-v1-20260909'


@dataclass(frozen=True)
class BattleEquityConfig:
    samples:int=1
    seed:int|None=None
    schema:str=SCHEMA

    def __post_init__(self):
        if self.schema!=SCHEMA or type(self.samples)is not int or self.samples not in (1,2):
            raise ValueError('This experiment permits exact M1 or MC M2 only')
        if self.samples==1 and self.seed is not None:raise ValueError('M1 has no private shaping seed')
        if self.samples==2 and (type(self.seed)is not int or not 0<=self.seed<2**63):
            raise ValueError('M2 requires an explicit private seed')

    def to_dict(self):return asdict(self)

    @classmethod
    def parse(cls,value):
        if value is None:return cls()
        if isinstance(value,cls):return value
        if not isinstance(value,dict) or set(value)!={'samples','seed','schema'}:
            raise ValueError('Incomplete/unknown reward-estimator schema')
        return cls(**value)


def resolve_config(checkpoint=None,*,samples=None,seed=None,allow_transfer=False):
    saved=BattleEquityConfig.parse((checkpoint or {}).get('training_semantics',{}).get('reward_estimator'))
    count=saved.samples if samples is None else samples
    selected_seed=(saved.seed if seed is None and count==saved.samples else seed) if count==2 else None
    if count==1 and seed is not None:raise ValueError('Do not provide an unused M1 shaping seed')
    requested=BattleEquityConfig(count,selected_seed)
    if checkpoint is not None and requested!=saved and not allow_transfer:
        raise ValueError('Reward-estimator transfer requires an explicit separate branch')
    return requested


def reward_table(core):
    """Use the current exact outcome arithmetic, including turn2 healing/caps."""
    lives=(core.agent.lives,core.opponent.lives);wins=(core.agent.wins,core.opponent.wins)
    values=[]
    for outcome in (0,1,2):
        scoreboard=SimpleNamespace(turn=core.turn,MAX_LIVES=core.MAX_LIVES,
            agent=SimpleNamespace(lives=lives[0],wins=wins[0]),
            opponent=SimpleNamespace(lives=lives[1],wins=wins[1]))
        type(core)._player_fight_outcome(scoreboard,outcome)
        values.append(0.1*((scoreboard.agent.lives-lives[0])-(scoreboard.opponent.lives-lives[1])))
    return tuple(values)


class BattleEquityEstimator:
    def __init__(self,config,owner):
        self.config=BattleEquityConfig.parse(config)
        if self.config.samples!=2 or not isinstance(owner,str) or not owner:raise ValueError('Explicit M2 owner required')
        self.owner=owner;self.generation=-1;self._rng=None;self.last=None

    def _seed(self,domain):
        text=f'{SCHEMA}:{self.config.seed}:{self.owner}:{self.generation}:{domain}'
        return int.from_bytes(hashlib.sha256(text.encode()).digest(),'big')

    def reset(self):
        self.generation+=1;self._rng=random.Random(self._seed('match'));self.last=None

    def redeterminize(self,seed):
        if self.generation<0 or type(seed)is not int:raise ValueError('Active match and integer future seed required')
        self._rng=random.Random(self._seed('future:'+str(seed)));self.last=None

    def prepare(self,core):
        if self._rng is None:raise ValueError('Reset before sampling reward equity')
        # Detach player/shop ownership. Normal Team.copy intentionally retains it.
        teams=tuple(Team.from_state(copy.deepcopy(player.team.state)) for player in (core.agent,core.opponent))
        if any(team.player is not None or team.shop is not None for team in teams):raise ValueError('Counterfactual teams retain live owners')
        return teams,reward_table(core)

    def estimate(self,prepared,actual_outcome,actual_component):
        teams,table=prepared
        if actual_outcome not in (0,1,2) or actual_component!=table[actual_outcome]:
            raise ValueError('Exact actual life-swing accounting changed')
        seed=self._rng.randrange(2**32-1) # with replacement, same valid Battle seed domain
        extra=int(Battle(*teams,seed=seed,fast_training=True).battle())
        if extra not in (0,1,2):raise ValueError('Invalid counterfactual battle outcome')
        shadow=table[extra]
        mean=actual_component if shadow==actual_component else (actual_component+shadow)/2
        self.last={'actualOutcome':int(actual_outcome),'extraOutcome':extra,'extraSeed':seed,
                   'actualComponent':actual_component,'extraComponent':shadow,'meanComponent':mean}
        return mean

    def state_dict(self):
        return {'config':self.config.to_dict(),'owner':self.owner,'generation':self.generation,
                'rng':None if self._rng is None else self._rng.getstate()}

    def restore_state(self,state):
        if (not isinstance(state,dict) or set(state)!={'config','owner','generation','rng'}
                or BattleEquityConfig.parse(state['config'])!=self.config or state['owner']!=self.owner
                or type(state['generation'])is not int or state['generation']< -1
                or (state['generation']==-1)!=(state['rng'] is None)):
            raise ValueError('Reward RNG ownership/schema mismatch')
        def tuples(x):return tuple(tuples(v) for v in x) if isinstance(x,(list,tuple)) else x
        restored=None
        if state['rng'] is not None:
            restored=random.Random(0)
            try:restored.setstate(tuples(state['rng']))
            except (ValueError,TypeError,OverflowError) as error:raise ValueError('Invalid reward RNG state') from error
        self.generation=state['generation'];self._rng=restored;self.last=None
