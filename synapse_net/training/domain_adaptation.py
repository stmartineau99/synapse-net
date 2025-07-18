import os
from typing import Optional, Tuple

import torch
import torch_em
import torch_em.self_training as self_training

from .semisupervised_training import get_unsupervised_loader
from .supervised_training import get_2d_model, get_3d_model, get_supervised_loader, _determine_ndim

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
    train_mask_paths: Optional[Tuple[str]] = None,
    val_mask_paths: Optional[Tuple[str]] = None,
    patch_sampler: Optional[callable] = None,
    pseudo_label_sampler: Optional[callable] = None,
    device: int = 0,
) -> None:
    """Run domain adaptation to transfer a network trained on a source domain for a supervised
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
        train_mask_paths: Boundary masks used by the patch sampler to accept or reject patches for training. 
        val_mask_paths: Sample masks used by the patch sampler to accept or reject patches for validation. 
        patch_sampler: Accept or reject patches based on a condition.
        pseudo_label_sampler: Mask out regions of the pseudo labels where the teacher is not confident before updating the gradients. 
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
    pseudo_labeler = self_training.DefaultPseudoLabeler(confidence_threshold=confidence_threshold)
    loss = self_training.DefaultSelfTrainingLoss()
    loss_and_metric = self_training.DefaultSelfTrainingLossAndMetric()
    
    unsupervised_train_loader = get_unsupervised_loader(
        data_paths=unsupervised_train_paths, 
        raw_key=raw_key, 
        patch_shape=patch_shape, 
        batch_size=batch_size, 
        n_samples=n_samples_train, 
        sample_mask_paths=train_mask_paths, 
        sampler=patch_sampler
    )
    unsupervised_val_loader = get_unsupervised_loader(
        data_paths=unsupervised_val_paths, 
        raw_key=raw_key, 
        patch_shape=patch_shape, 
        batch_size=batch_size, 
        n_samples=n_samples_val, 
        sample_mask_paths=val_mask_paths, 
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
