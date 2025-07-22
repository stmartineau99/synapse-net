import os
from typing import Optional, Tuple

import torch
import torch_em
import torch_em.self_training as self_training

from .semisupervised_training import get_unsupervised_loader
from .supervised_training import get_2d_model, get_3d_model, get_supervised_loader, _determine_ndim

class NewPseudoLabeler(self_training.DefaultPseudoLabeler):
    """Compute pseudo labels based on model predictions, typically from a teacher model.
        By default, assumes that the first channel contains the transformed data and the second channel contains the background mask. # TODO update description

    Args:
        activation: Activation function applied to the teacher prediction.
        confidence_threshold: Threshold for computing a mask for filtering the pseudo labels.
            If None is given no mask will be computed.
        threshold_from_both_sides: Whether to include both values bigger than the threshold
            and smaller than 1 - the thrhesold, or only values bigger than the threshold, in the mask.
            The former should be used for binary labels, the latter for for multiclass labels.
        confidence_mask_channel: A specific channel to use for computing the confidence mask.
            By default the confidence mask is computed across all channels independently.
            This is useful, if only one of the channels encodes a probability.
        raw_channel: # TODO add description
        background_mask_channel: # TODO add description
    """
    def __init__(
        self,
        activation: Optional[torch.nn.Module] = None,
        confidence_threshold: Optional[float] = None,
        threshold_from_both_sides: bool = True,
        confidence_mask_channel: Optional[int] = None,
        raw_channel: Optional[int] = 0, 
        background_mask_channel: Optional[int] = 1,
    ):
        super().__init__(activation, confidence_threshold, threshold_from_both_sides)
        self.raw_channel = raw_channel
        self.background_mask_channel = background_mask_channel
        self.confidence_mask_channel = confidence_mask_channel
    
    def _subtract_background(self, pseudo_labels: torch.Tensor, background_mask: torch.Tensor):
        bool_mask = background_mask.bool()
        return pseudo_labels.masked_fill(bool_mask, 0)

    def __call__(self, teacher: torch.nn.Module, input_: torch.Tensor) -> torch.Tensor:
        """Compute pseudo-labels.

        Args:
            teacher: The teacher model.
            input_: The input for this batch.

        Returns:
            The pseudo-labels.
        """
        if self.background_mask_channel is not None:
            if input_.ndim != 5:
                raise ValueError(f"Expect data with 5 dimensions (B, C, D, H, W), got shape {input_.shape}.") 
    
            if self.background_mask_channel > input_.shape[1]:
                raise ValueError(f"Channel index {self.background_mask_channel} is out of bounds for shape {input_.shape}.")

            background_mask = input_[:, self.background_mask_channel].unsqueeze(1)
            input_ = input_[:, self.raw_channel].unsqueeze(1)

        pseudo_labels = teacher(input_)

        if self.activation is not None:
            pseudo_labels = self.activation(pseudo_labels)
        if self.confidence_threshold is None:
            label_mask = None
        else:
            mask_input = pseudo_labels if self.confidence_mask_channel is None\
                else pseudo_labels[self.confidence_mask_channel:(self.confidence_mask_channel+1)]
            label_mask = self._compute_label_mask_both_sides(mask_input) if self.threshold_from_both_sides\
                else self._compute_label_mask_one_side(mask_input)
            if self.confidence_mask_channel is not None:
                size = (pseudo_labels.shape[0], pseudo_labels.shape[1], *([-1] * (pseudo_labels.ndim - 2)))
                label_mask = label_mask.expand(*size)
        
        if self.background_mask_channel is not None:  
            pseudo_labels = self._subtract_background(pseudo_labels, background_mask)

        return pseudo_labels, label_mask


def mean_teacher_adaptation(
    name: str,
    unsupervised_train_paths: Tuple[str],
    unsupervised_val_paths: Tuple[str],
    patch_shape: Tuple[int, int, int],
    save_root: Optional[str] = None,
    source_checkpoint: Optional[str] = None,
    supervised_train_paths: Optional[Tuple[str]] = None,
    supervised_val_paths: Optional[Tuple[str]] = None,
    confidence_threshold: float = 0.9,
    raw_key: str = "raw",
    raw_key_supervised: str = "raw",
    label_key: Optional[str] = None,
    batch_size: int = 1,
    lr: float = 1e-4,
    n_iterations: int = int(1e4),
    n_samples_train: Optional[int] = None,
    n_samples_val: Optional[int] = None,
    train_sample_mask_paths: Optional[Tuple[str]] = None,
    val_sample_mask_paths: Optional[Tuple[str]] = None,
    train_background_mask_paths: Optional[Tuple[str]] = None,
    patch_sampler: Optional[callable] = None,
    pseudo_label_sampler: Optional[callable] = None,
    device: int = 0,
) -> None:
    """Run domain adapation to transfer a network trained on a source domain for a supervised
    segmentation task to perform this task on a different target domain.

    We support different domain adaptation settings:
    - unsupervised domain adaptation: the default mode when 'supervised_train_paths' and
     'supervised_val_paths' are not given.
    - semi-supervised domain adaptation: domain adaptation on unlabeled and labeled data,
      when 'supervised_train_paths' and 'supervised_val_paths' are given.

    Args:
        name: The name for the checkpoint to be trained.
        unsupervsied_train_paths: Filepaths to the hdf5 files or similar file formats
            for the training data in the target domain.
            This training data is used for unsupervised learning, so it does not require labels.
        unsupervised_val_paths: Filepaths to the hdf5 files or similar file formats
            for the validation data in the target domain.
            This validation data is used for unsupervised learning, so it does not require labels.
        patch_shape: The patch shape used for a training example.
            In order to run 2d training pass a patch shape with a singleton in the z-axis,
            e.g. 'patch_shape = [1, 512, 512]'.
        save_root: Folder where the checkpoint will be saved.
        source_checkpoint: Checkpoint to the initial model trained on the source domain.
            This is used to initialize the teacher model.
            If the checkpoint is not given, then both student and teacher model are initialized
            from scratch. In this case `supervised_train_paths` and `supervised_val_paths` have to
            be given in order to provide training data from the source domain.
        supervised_train_paths: Filepaths to the hdf5 files for the training data in the source domain.
            This training data is optional. If given, it is used for unsupervised learnig and requires labels.
        supervised_val_paths: Filepaths to the df5 files for the validation data in the source domain.
            This validation data is optional. If given, it is used for unsupervised learnig and requires labels.
        confidence_threshold: The threshold for filtering data in the unsupervised loss.
            The label filtering is done based on the uncertainty of network predictions, and only
            the data with higher certainty than this threshold is used for training.
        raw_key: The key that holds the raw data inside of the hdf5 or similar files.
        label_key: The key that holds the labels inside of the hdf5 files for supervised learning.
            This is only required if `supervised_train_paths` and `supervised_val_paths` are given.
        batch_size: The batch size for training.
        lr: The initial learning rate.
        n_iterations: The number of iterations to train for.
        n_samples_train: The number of train samples per epoch. By default this will be estimated
            based on the patch_shape and size of the volumes used for training.
        n_samples_val: The number of val samples per epoch. By default this will be estimated
            based on the patch_shape and size of the volumes used for validation.
        train_sample_mask_paths: Boundary masks used by the patch sampler to accept or reject patches for training. 
        val_sample_mask_paths: Sample masks used by the patch sampler to accept or reject patches for validation. 
        train_background_mask_paths: # TODO add description
        patch_sampler: A sampler for rejecting patches based on a defined conditon. 
        pseudo_label_sampler: A sampler for rejecting pseudo-labels based on a defined condition.
        device: GPU ID for training. 
    """
    assert (supervised_train_paths is None) == (supervised_val_paths is None)
    is_2d, _ = _determine_ndim(patch_shape)

    if source_checkpoint is None:
        # training from scratch only makes sense if we have supervised training data
        # that's why we have the assertion here.
        assert supervised_train_paths is not None
        print("Mean teacher training from scratch (AdaMT)")
        if is_2d:
            model = get_2d_model(out_channels=2)
        else:
            model = get_3d_model(out_channels=2)
        reinit_teacher = True
    else:
        print("Mean teacher training initialized from source model:", source_checkpoint)
        if os.path.isdir(source_checkpoint):
            model = torch_em.util.load_model(source_checkpoint)
        else:
            model = torch.load(source_checkpoint)
        reinit_teacher = False

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    # self training functionality
    if train_background_mask_paths is not None:
        pseudo_labeler = NewPseudoLabeler(confidence_threshold=confidence_threshold, background_mask_channel=1)
    else:
        pseudo_labeler = self_training.DefaultPseudoLabeler(confidence_threshold=confidence_threshold)

    loss = self_training.DefaultSelfTrainingLoss()
    loss_and_metric = self_training.DefaultSelfTrainingLossAndMetric()
   
    unsupervised_train_loader = get_unsupervised_loader(
        data_paths=unsupervised_train_paths, 
        raw_key=raw_key, 
        patch_shape=patch_shape, 
        batch_size=batch_size, 
        n_samples=n_samples_train, 
        sample_mask_paths=train_sample_mask_paths, 
        background_mask_paths=train_background_mask_paths,
        sampler=patch_sampler
    )
    unsupervised_val_loader = get_unsupervised_loader(
        data_paths=unsupervised_val_paths, 
        raw_key=raw_key, 
        patch_shape=patch_shape, 
        batch_size=batch_size, 
        n_samples=n_samples_val, 
        sample_mask_paths=val_sample_mask_paths, 
        background_mask_paths=None,
        sampler=patch_sampler
    )

    if supervised_train_paths is not None:
        assert label_key is not None
        supervised_train_loader = get_supervised_loader(
            supervised_train_paths, raw_key_supervised, label_key,
            patch_shape, batch_size, n_samples=n_samples_train,
        )
        supervised_val_loader = get_supervised_loader(
            supervised_val_paths, raw_key_supervised, label_key,
            patch_shape, batch_size, n_samples=n_samples_val,
        )
    else:
        supervised_train_loader = None
        supervised_val_loader = None

    device = torch.device(f"cuda:{device}") if torch.cuda.is_available() else torch.device("cpu")
    trainer = self_training.MeanTeacherTrainer(
        name=name,
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        pseudo_labeler=pseudo_labeler,
        unsupervised_loss=loss,
        unsupervised_loss_and_metric=loss_and_metric,
        supervised_train_loader=supervised_train_loader,
        unsupervised_train_loader=unsupervised_train_loader,
        supervised_val_loader=supervised_val_loader,
        unsupervised_val_loader=unsupervised_val_loader,
        supervised_loss=loss,
        supervised_loss_and_metric=loss_and_metric,
        logger=self_training.SelfTrainingTensorboardLogger,
        mixed_precision=True,
        log_image_interval=100,
        compile_model=False,
        device=device,
        reinit_teacher=reinit_teacher,
        save_root=save_root,
        sampler=pseudo_label_sampler,
    )
    trainer.fit(n_iterations)
