import json

from logzero import logger


class TrainingScheduler:
    def __init__(
            self,
            start_epoch: int = 0,
            max_epoch: int = 150,
            early_stopping_thres: int = 5
    ):
        self.epoch = start_epoch
        self.max_epoch = max_epoch
        self.early_stopping_thres = early_stopping_thres

        self.early_stopping_count: int = 0
        self.best_epoch: int = -1
        self.best_performance: float = -1
        self.performances: [float, ...] = []

    def __call__(self, performance: float) -> (bool, bool, bool):
        """
        Args:
            * performance: The model performance
        Returns:
            * is_best: Whether performance is the best
            * is_over_thres: Whether to reach the threshold of early stopping
            * is_max_epoch: Whether to reach the maximum epoch
        """
        is_best = False
        is_over_thres = False
        is_max_epoch = False

        self.early_stopping_count += 1
        self.epoch += 1
        self.performances.append(performance)
        assert len(self.performances) == self.epoch

        # The case of updating the best performance
        if performance > self.best_performance:
            logger.info(f"Update best performance: {self.best_performance} -> {performance}")
            self.early_stopping_count = 0
            self.best_epoch = self.epoch
            self.best_performance = performance
            is_best = True

        # The early stopping count exceeds the thresh of early stopping
        elif self.early_stopping_thres != 0 and self.early_stopping_count >= self.early_stopping_thres:
            self.early_stopping_count = 0
            is_over_thres = True

        if self.epoch >= self.max_epoch:
            logger.info("Finish training. Epoch reaches the maximum.")
            logger.info(f"Best performance: {self.best_performance} (epoch={self.best_epoch})")
            is_max_epoch = True

            if self.epoch != self.max_epoch:
                logger.warning(
                    f"WARNING: the epoch is already over ({self.epoch} > {self.max_epoch})! "
                    "you should check the training code."
                )

        return is_best, is_over_thres, is_max_epoch

    def __repr__(self):
        return json.dumps(self.__dict__)

    def save(self, file_path):
        with open(file_path, "w") as fo:
            json.dump(self.__dict__, fo)
        logger.info(f"Save: {file_path}")

    def load(self, file_path):
        with open(file_path) as fi:
            self.__dict__ = json.load(fi)
        logger.info(f"Load: {file_path}")


class MyLRScheduler:
    def __init__(
            self,
            optimizer_states: dict,
            start_lr: float,
            min_lr: float,
            ratio_reduce_lr: float = 0.5
    ):
        self.optimizer_states = optimizer_states
        self.lr = start_lr
        self.min_lr = min_lr
        self.ratio_reduce_lr = ratio_reduce_lr
        assert self.min_lr != 0
        assert self.ratio_reduce_lr != 0

    def get_state(self):
        if self.lr > self.min_lr + self.min_lr * 1e-4:
            new_lr = max(self.lr * self.ratio_reduce_lr, self.min_lr)
            logger.info(f"Update learning rate: {self.lr} -> {new_lr}")
            self.lr = new_lr
            self.optimizer_states["lr"] = new_lr

            return self.optimizer_states

        else:
            logger.info("Learning rate has reached the minimum learning rate.")

            return None
