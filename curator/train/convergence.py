"""Early-stopping policy; scheduling, metrics and persistence belong elsewhere."""
from collections import deque
from statistics import median

import torch
from pytorch_lightning.callbacks import EarlyStopping


class LearningRateGatedEarlyStopping(EarlyStopping):
    """Apply Lightning's stopping decision after low-LR physical plateaus.

    ``physical_metrics`` must name finite, nonnegative validation metrics logged
    in the current epoch. No metrics file is read and no optimizer is modified.
    Safety/explicit-threshold stops retain Lightning's normal behavior.
    """

    def __init__(self, monitor, physical_metrics=(), min_lr=1e-6,
                 min_lr_checks=12, relative_min_delta=0.005, plateau_window=6,
                 plateau_rtol=0.005, deterioration_rtol=0.02, **kwargs):
        if kwargs.get('check_on_train_epoch_end', False):
            raise ValueError('Use validation-end checks after physical metrics are logged')
        kwargs['check_on_train_epoch_end'] = False
        if relative_min_delta is not None and kwargs.get('min_delta', 0.0) != 0.0:
            raise ValueError('Choose relative_min_delta or absolute min_delta, not both')
        super().__init__(monitor=monitor, **kwargs)
        if min_lr <= 0 or min_lr_checks < 1 or plateau_window < 1:
            raise ValueError('LR floor and observation windows must be positive')
        if relative_min_delta is not None and not 0 <= relative_min_delta < 1:
            raise ValueError('relative_min_delta must be in [0, 1)')
        if not 0 <= plateau_rtol < 1 or deterioration_rtol < 0:
            raise ValueError('Invalid plateau tolerances')
        self.physical_metrics = tuple(physical_metrics)
        self.min_lr = float(min_lr)
        self.min_lr_checks = int(min_lr_checks)
        self.relative_min_delta = relative_min_delta
        self.plateau_window = int(plateau_window)
        self.plateau_rtol = float(plateau_rtol)
        self.deterioration_rtol = float(deterioration_rtol)
        self.history = deque(maxlen=2 * self.plateau_window)
        self.last_lrs = None
        self.low_lr_checks = 0
        self._gate_open = False

    def _run_early_stopping_check(self, trainer):
        if trainer.fast_dev_run or not self._validate_condition_metric(trainer.callback_metrics):
            return
        lrs = tuple(g['lr'] for opt in trainer.optimizers for g in opt.param_groups)
        if not lrs:
            raise RuntimeError('Early stopping requires an optimizer')
        values = []
        for key in self.physical_metrics:
            if key not in trainer.callback_metrics:
                raise RuntimeError(f'Missing physical validation metric: {key}')
            value = float(trainer.callback_metrics[key])
            if not torch.isfinite(torch.tensor(value)) or value < 0:
                raise ValueError(f'Physical metric {key} must be finite and nonnegative')
            values.append(value)
        if lrs != self.last_lrs:
            self.low_lr_checks = 0
            self.history.clear()
        self.last_lrs = lrs
        self.low_lr_checks = self.low_lr_checks + 1 if max(lrs) <= self.min_lr * (1 + 1e-8) else 0
        self.history.append(values)
        self._gate_open = self.low_lr_checks >= self.min_lr_checks and self._physical_plateau()
        super()._run_early_stopping_check(trainer)

    def _physical_plateau(self):
        if not self.physical_metrics:
            return True
        if len(self.history) < self.history.maxlen:
            return False
        rows = list(self.history)
        for index in range(len(self.physical_metrics)):
            before = median(row[index] for row in rows[:self.plateau_window])
            after = median(row[index] for row in rows[self.plateau_window:])
            if before == 0.0:
                if after != 0.0:
                    return False
            elif not 1 - self.plateau_rtol <= after / before <= 1 + self.deterioration_rtol:
                return False
        return True

    def _evaluate_stopping_criteria(self, current):
        if self.relative_min_delta is not None:
            delta = abs(float(self.best_score)) * self.relative_min_delta if torch.isfinite(self.best_score) else 0.0
            self.min_delta = delta if self.mode == 'max' else -delta
        stop, reason = super()._evaluate_stopping_criteria(current)
        safety_or_threshold = (
            (self.check_finite and not torch.isfinite(current))
            or (self.stopping_threshold is not None and self.monitor_op(current, self.stopping_threshold))
            or (self.divergence_threshold is not None and self.monitor_op(-current, -self.divergence_threshold))
        )
        return (stop, reason) if safety_or_threshold or self._gate_open else (False, None)

    def state_dict(self):
        return {**super().state_dict(), 'history': list(self.history),
                'last_lrs': self.last_lrs, 'low_lr_checks': self.low_lr_checks}

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.history = deque(state_dict['history'], maxlen=2 * self.plateau_window)
        self.last_lrs = None if state_dict['last_lrs'] is None else tuple(state_dict['last_lrs'])
        self.low_lr_checks = state_dict['low_lr_checks']
