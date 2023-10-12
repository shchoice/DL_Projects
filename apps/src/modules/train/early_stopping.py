import os
import shutil

from transformers import EarlyStoppingCallback, Trainer


class SaveLastModelCallback(EarlyStoppingCallback):
    def __init__(self, early_stopping_patience: int = 0):
        super().__init__(early_stopping_patience=early_stopping_patience)
        self.trainer = None

    def set_trainer(self, trainer: Trainer):
        self.trainer = trainer

    def on_train_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            last_checkpoint = os.path.join(args.output_dir, "checkpoint-last")
            self.trainer.save_model(last_checkpoint)

            # early stopping이 동작하지 않은 학습의 경우 best_checkpoint가 없기에 last 만 저장하고 패스
            best_checkpoint = self.trainer.state.best_model_checkpoint
            if best_checkpoint:
                final_path = os.path.join(os.path.dirname(best_checkpoint), "checkpoint-best")
                if os.path.exists(final_path):
                    shutil.rmtree(final_path)
                shutil.copytree(best_checkpoint, final_path)
