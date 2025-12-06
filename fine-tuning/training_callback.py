import time
import wandb
from transformers import TrainerCallback, EarlyStoppingCallback

class TokenSpeedCallback(TrainerCallback):
    def __init__(self, seq_len, batch_size, grad_accum):
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.grad_accum = grad_accum
        self.last_time = None
        self.last_step = None

    def on_step_begin(self, args, state, control, **kwargs):
        if self.last_time is None:
            self.last_time = time.time()
            self.last_step = state.global_step

    def on_step_end(self, args, state, control, **kwargs):
        now = time.time()

        step_delta = state.global_step - self.last_step
        if step_delta <= 0:
            return

        dt = now - self.last_time
        # tokens per optimizer *step*
        tokens_per_step = self.seq_len * self.batch_size * self.grad_accum
        # total tokens processed since last measurement
        tokens_processed = tokens_per_step * step_delta

        tps = tokens_processed / dt  # tokens per second

        wandb.log({
            "tokens_per_second": tps,
            "total_tokens_seen": state.global_step * tokens_per_step,
        })

        self.last_time = now
        self.last_step = state.global_step

early_stop = EarlyStoppingCallback(
    early_stopping_patience=3,   # stop after 3 evals with no improvement
    early_stopping_threshold=0.0 # min_delta for improvement (0 = any improvement)
)
