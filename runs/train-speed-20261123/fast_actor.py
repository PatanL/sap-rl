"""Keep the native distributions and sampler, but move legality/finite assertions off the host critical path."""
import torch
from torch.distributions import Categorical
from sapai_gym1.rl.end_freeze import EndFreezeHead,SHOP_ACTION_CAP
from sapai_gym1.rl.action_schema import ActionType,MAX_SHOP_SLOTS,NO_TARGET
from sapai_gym1.rl.model import EntityTransformerPolicy
class AsyncCheckedPolicy(EntityTransformerPolicy):
    @staticmethod
    def _masked_logits(logits,mask):
        torch._assert_async(mask.any(-1).all(),'No legal autoregressive choice')
        masked=logits.masked_fill(~mask.bool(),torch.finfo(logits.dtype).min)
        torch._assert_async(torch.isfinite(masked).all(),'Invalid policy logits')
        return masked
class AsyncCheckedFreeze(EndFreezeHead):
    def distribution(self,slot_logits,observation,action_types):
        if slot_logits.ndim!=2 or slot_logits.shape[1]!=MAX_SHOP_SLOTS:raise ValueError('Wrong physical slot count')
        torch._assert_async(torch.isfinite(slot_logits).all(),'Nonfinite freeze logits')
        mask=observation['target_mask'].bool()
        eligible=mask[:,ActionType.FREEZE,:MAX_SHOP_SLOTS,NO_TARGET] | mask[:,ActionType.UNFREEZE,:MAX_SHOP_SLOTS,NO_TARGET]
        active=(action_types==int(ActionType.END_TURN)) & (observation['numeric'][:,0,11].abs()<1e-8) & observation['type_mask'][:,ActionType.END_TURN].bool()
        eligible=eligible & active[:,None]
        remaining=(SHOP_ACTION_CAP-(observation['numeric'][:,0,5].float()*SHOP_ACTION_CAP).round().long()).clamp(0,MAX_SHOP_SLOTS)
        with torch.autocast(slot_logits.device.type,enabled=False):
            illegal=(~eligible).float() @ self.fast_bits.T > 0
            allowed=(~illegal) & (self.subset_sizes[None]<=remaining[:,None])
            logits=(slot_logits.float() @ self.fast_bits.T).masked_fill(~allowed,-torch.inf)
            return Categorical(logits=logits,validate_args=False),allowed

def enable(actor):
    if type(actor) is not EntityTransformerPolicy:raise TypeError('Unexpected actor class')
    actor.__class__=AsyncCheckedPolicy
    head=actor.end_freeze_head
    if head is not None:
        head.__class__=AsyncCheckedFreeze;head.register_buffer('fast_bits',head.subsets.float(),persistent=False)
    return actor
