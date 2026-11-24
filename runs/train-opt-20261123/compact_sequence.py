"""Shadows runs/train-speed-20261123/compact_sequence.py on purpose (this directory is first on PYTHONPATH): the same
function, plus an optional single cuDNN call for the recurrence (OPT_CUDNN_GRU=1). Without that variable it is unchanged."""
import os
import torch
from sapai_gym1.rl.model import EncodedDecision,SequencePolicyOutput

def evaluate(policy,p,layout):
    flat_ix,lookup,history_constant=layout
    lengths=p.actions.shape[:2];t,b=lengths
    flat={k:v.flatten(0,1).index_select(0,flat_ix) for k,v in p.observations.items()}
    tok=policy.encode_entities(flat)
    h=p.initial_memories;states=[]
    first=tok[:,0].index_select(0,lookup.flatten()).view(t,b,-1)
    if os.environ.get('OPT_CUDNN_GRU')=='1':
        # The whole recurrence in one cuDNN call instead of one GRUCell call per time step. nn.GRUCell and a
        # one-layer nn.GRU use the same equations and the same parameter layout, so the cell's own weights are
        # passed as they are. Run in FP32 (the recurrence is tiny; this is more exact than the BF16 cell).
        cell=policy.memory_cell
        with torch.autocast('cuda',enabled=False):
            states,last=torch._VF.gru(first.float(),h.float()[None],(cell.weight_ih,cell.weight_hh,cell.bias_ih,cell.bias_hh),True,1,0.,policy.training,False,False)
        h=last[0]
    else:
        for x in first.unbind():
            h=policy.memory_cell(x,h);states.append(h)
        states=torch.stack(states)
    memory=states.flatten(0,1).index_select(0,flat_ix)
    ctx=policy.context_projection(torch.cat((tok[:,0],memory),dim=-1))
    if history_constant:
        belief=policy.encode_opponent_history({k:v[0] for k,v in p.observations.items()})
        adapter=policy.opponent_belief_adapter(belief)
        ctx=ctx+adapter.index_select(0,flat_ix%b)
        prediction=policy.predict_opponent_board(belief)
    else:
        belief=policy.encode_opponent_history(flat)
        ctx=ctx+policy.opponent_belief_adapter(belief)
        prediction=policy.predict_opponent_board(belief[:b])
    if policy.config.causal_context_schema is not None:raise ValueError('Compact path only qualified for public 25,772')
    # The native decoder receives its teacher actions and genuine joint masks; only t=0 forecasts enter the loss.
    features=EncodedDecision(flat,tok,ctx,memory,prediction,belief)
    out=policy.decode_decision(features,actions=p.actions.flatten(0,1).index_select(0,flat_ix),
            allow_end_freeze=True,end_freeze_masks=p.freeze_masks.flatten().index_select(0,flat_ix))
    def pad(v):return v.index_select(0,lookup.flatten()).reshape(t,b,*v.shape[1:])
    return SequencePolicyOutput(log_prob=pad(out.log_prob),entropy=pad(out.entropy),value=pad(out.value),
        memory_states=states,final_memory=h,battle_logits=pad(out.battle_logits),board_strength=pad(out.board_strength),
        **{k:v.unsqueeze(0) for k,v in prediction.items()})
