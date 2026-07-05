use std::f32::consts::PI;

#[derive(Debug, PartialEq, Copy, Clone)]
pub struct CosineDecayLR {
    min_lr: f32,
    max_lr: f32,
    cur_lr: f32,
    cur_step: usize,
    total_step: usize,
    warmup_steps: usize
}

impl CosineDecayLR {
    /// Creates a new instance of this scheduler.
    ///
    /// # Arguments
    /// * `min_lr` - Minimum target learning rate.
    /// * `max_lr` - Maximum peak learning rate.
    /// * `total_step` - Total target steps
    /// * `warmup_steps` - Steps of increasing learning rate to warm up the network
    /// rate is called.
    pub fn new(min_lr: f32, max_lr: f32, total_step: usize, warmup_steps: usize) -> Self {
        let mut scheduler = Self {
            min_lr,
            max_lr,
            cur_lr: min_lr,
            cur_step: 0,
            total_step,
            warmup_steps
        };

        scheduler.update_lr();
        scheduler
    }

    fn update_lr(&mut self) {
        if self.cur_step < self.warmup_steps {
            let progress = self.cur_step as f32 / self.warmup_steps as f32;
            self.cur_lr = self.min_lr + progress * (self.max_lr - self.min_lr);
        } else {
            let decay_step = self.cur_step - self.warmup_steps;
            let decay_total = self.total_step - self.warmup_steps;

            if decay_total == 0 {
                self.cur_lr = self.min_lr;
                return;
            }

            let progress = decay_step as f32 / decay_total as f32;
            self.cur_lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1.0 + (progress * PI).cos());
        }
    }

    /// Resets the scheduler back into the original learning rate
    pub fn reset(&mut self) {
        self.cur_step = 0;
        self.update_lr();
    }

    /// Returns the current learning rate value.
    pub fn get_learning_rate(&self) -> f32 {
        self.cur_lr
    }

    /// Expected to be called once within a step.
    ///
    /// # Returns
    /// The updated learning rate.
    pub fn step(&mut self) -> f32 {
        self.cur_step += 1;
        self.update_lr();
        self.cur_lr
    }
}

/// This learning rate scheduler focuses on decreasing the learning rate exponentially.
#[derive(Debug, PartialEq, Copy, Clone)]
pub struct ExponentialLR {
    ori_lr: f32,
    cur_lr: f32,
    lr_factor: f32,
}

impl ExponentialLR {
    /// Creates a new instance of this scheduler.
    ///
    /// # Arguments
    /// * `lr_initial` - The initial learning rate.
    /// * `lr_factor` - A factor multiplied with the current learning factor when a change in learning
    /// rate is called.
    pub fn new(lr_initial: f32, lr_factor: f32) -> Self {
        Self {
            ori_lr: lr_initial,
            cur_lr: lr_initial,
            lr_factor
        }
    }

    /// Resets the scheduler back into the original learning rate
    pub fn reset(&mut self) {
        self.cur_lr = self.ori_lr;
    }


    /// Returns the current learning rate value.
    pub fn get_learning_rate(&self) -> f32 {
        self.cur_lr
    }

    /// Expected to be called once within a step.
    ///
    /// # Returns
    /// The updated learning rate.
    pub fn step(&mut self) -> f32 {
        self.cur_lr *= self.lr_factor;
        self.cur_lr
    }
}

/// This learning rate scheduler focuses on decreasing the learning rate when there is no improvement
/// on the test value (e.g. test accuracy) over a period of steps.
#[derive(Debug, PartialEq, Copy, Clone)]
pub struct ReduceLROnPlateauScheduler {
    best_val: f32,
    mode: SchedulerMode,
    min_lr: f32,
    cur_lr: f32,
    lr_factor: f32,
    timer: usize,
    patience: usize
}

/// Used for determining the improvement direction for a given test value in the schedulers.
#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub enum SchedulerMode {
    /// Mainly used for loss values.
    Minimize,
    /// Mainly used for accuracy values.
    Maximize
}

impl ReduceLROnPlateauScheduler {
    /// Creates a new instance of this scheduler.
    ///
    /// # Arguments
    /// * `patience` - The maximum amount of non-improvement steps from the network before
    /// changing the learning rate.
    /// * `mode` - The scheduler mode. See [`SchedulerMode`].
    /// * `lr_factor` - A factor multiplied with the current learning factor when a change in learning
    /// rate is called.
    /// * `min_lr` - The minimum learning rate value. When the scheduler reaches this value,
    /// the learning rate will stay here for the rest of the steps.
    ///
    /// # Panics
    /// Panics if `patience` is set to `0` since a `patience` of `0` means the learning rate never changes.
    pub fn new(patience: usize, mode: SchedulerMode, lr_factor: f32, min_lr: f32) -> Self {
        assert_ne!(patience, 0, "Patience must be non-zero.");
        
        Self {
            best_val: if mode == SchedulerMode::Maximize { f32::MIN } else { f32::MAX },
            mode,
            min_lr,
            cur_lr: 0.0,
            lr_factor,
            timer: 0,
            patience
        }
    }

    /// Resets the scheduler.
    ///
    /// # Arguments
    /// * `learning_rate` - The learning rate value to reset into.
    pub fn reset(&mut self, learning_rate: f32) {
        self.cur_lr = learning_rate;
        self.timer = 0;
        self.best_val = if self.mode == SchedulerMode::Maximize { f32::MIN } else { f32::MAX };
    }

    /// Returns the current learning rate value.
    pub fn get_learning_rate(&self) -> f32 {
        self.cur_lr
    }

    /// Expected to be called once within a step.
    ///
    /// # Arguments
    /// * `test_val` - The test value itself.
    ///
    /// # Returns
    /// The updated learning rate.
    pub fn step(&mut self, test_val: f32) -> f32 {
        if test_val.is_nan() {
            return self.cur_lr;
        }

        let improved = match self.mode {
            SchedulerMode::Maximize => test_val > self.best_val,
            SchedulerMode::Minimize => test_val < self.best_val,
        };

        if improved {
            self.best_val = test_val;
            self.timer = 0;
        } else {
            self.timer += 1;
            if self.timer == self.patience {
                self.timer = 0;
                self.cur_lr *= self.lr_factor;
                self.cur_lr = self.cur_lr.max(self.min_lr);
            }
        }

        self.cur_lr
    }
}