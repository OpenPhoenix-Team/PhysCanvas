"""Bounded real-policy optimizer check, NOT a WM success/learning benchmark.

Eight real flow-SDE samples share a real reset. Fixed diagnostic returns (four
zeros/four ones) exercise the nonempty GRPO update path even when a short WM
smoke has no successes. Sequential replay bounds memory; this diagnostic is
not a reproduction of the production distributed/no-accumulation topology.
Only a JSON report is written; source weights are never saved or overwritten.
"""
import argparse
import json
from pathlib import Path
import torch
from .config import compose_config
from .backend import PhysCanvasBackend

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', choices=['jimu', 'shiguan'], required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    cfg = compose_config(a.task)
    torch.manual_seed(42)
    from rlinf.models.embodiment.openpi_rlinf import get_model
    from rlinf.envs.world_model.jimu_wm_protocol import stack_policy_observations
    from rlinf.algorithms.advantages import compute_grpo_advantages
    from rlinf.algorithms.losses import compute_ppo_actor_loss
    backend = PhysCanvasBackend(cfg.env.train.world_model, device='cuda:0', seed=42,
                                prompt=cfg.env.train.prompt, reset_start_index=0)
    reset = backend.reset_groups(num_groups=1, group_size=1, prompt=cfg.env.train.prompt)
    obs = stack_policy_observations(reset['observations'], prompt=cfg.env.train.prompt, policy_chunk_index=0)
    model = get_model(cfg.actor.model).cuda().eval()
    samples = []
    with torch.no_grad():
        for i in range(8):
            actions, info = model.predict_action_batch(obs, mode='train', compute_values=False)
            assert torch.isfinite(torch.as_tensor(actions)).all()
            samples.append((info['prev_logprobs'].detach().cpu(),
                {k:v.detach().cpu() for k,v in info['forward_inputs'].items()}))
            print(f'Sampled diagnostic candidate {i+1}/8', flush=True)
    advantages, _ = compute_grpo_advantages(torch.tensor([0.,0.,0.,0.,1.,1.,1.,1.]),
                                           torch.ones(1,8), group_size=8)
    trainable = [v for v in model.parameters() if v.requires_grad]
    before = [v.detach().float().cpu().clone() for v in trainable]
    opt = torch.optim.AdamW(trainable, lr=5e-6, betas=(.9,.95), eps=1e-8, weight_decay=.01)
    opt.zero_grad(set_to_none=True)
    losses, logprob_errors = [], []
    for i, (old, inputs) in enumerate(samples):
        result = model(forward_inputs={k:v.cuda() for k,v in inputs.items()}, compute_values=False)
        new = result['logprobs']
        old = old.cuda().reshape_as(new)
        logprob_errors.append((new.detach()-old).abs().max().item())
        advantage = torch.full_like(new, advantages[0,i].item())
        loss, _ = compute_ppo_actor_loss(new.float(), old.float(), .2, .28,
                                        advantage.float(), clip_ratio_c=3.)
        assert torch.isfinite(loss)
        (loss/8).backward()
        losses.append(loss.item())
        del result, new, loss
        print(f'Replayed diagnostic candidate {i+1}/8', flush=True)
    norm = torch.nn.utils.clip_grad_norm_(trainable,1.)
    assert torch.isfinite(norm) and norm > 0
    opt.step()
    changed = sum(not torch.equal(b, v.detach().float().cpu()) for b,v in zip(before,trainable))
    assert changed > 0
    report = {'task':a.task, 'diagnostic_returns':[0,0,0,0,1,1,1,1],
              'wm_reward_used':False, 'distributed_training_verified':False,
              'group_size':8,'optimizer_steps':1,'changed_parameter_tensors':changed,
              'gradient_norm':norm.item(),'losses':losses,
              'max_recomputed_logprob_errors':logprob_errors,
              'peak_gpu_bytes':torch.cuda.max_memory_allocated(), 'source_checkpoint_modified':False}
    (a.output/'result.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)
    backend.close()

if __name__ == '__main__':
    main()
