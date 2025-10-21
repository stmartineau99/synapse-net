import os
from glob import glob
from typing import Optional, Tuple

import torch
import torch_em
from sklearn.model_selection import train_test_split
from torch_em.model import AnisotropicUNet, UNet2d


def get_3d_model(
    out_channels: int,
    in_channels: int = 1,
    scale_factors: Tuple[Tuple[int, int, int]] = [[1, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
    initial_features: int = 32,
    final_activation: str = "Sigmoid",
) -> torch.nn.Module:
    """Get the U-Net model for 3D segmentation tasks.

    Args:
        out_channels: The number of output channels of the network.
        scale_factors: The downscaling factors for each level of the U-Net encoder.
        initial_features: The number of features in the first level of the U-Net.
            The number of features increases by a factor of two in each level.
        final_activation: The activation applied to the last output layer.

    Returns:
        The U-Net.
    """
    model = AnisotropicUNet(
        scale_factors=scale_factors,
        in_channels=in_channels,
        out_channels=out_channels,
        initial_features=initial_features,
        gain=2,
        final_activation=final_activation,
    )
    return model


def get_2d_model(
    out_channels: int,
    in_channels: int = 1,
    initial_features: int = 32,
    final_activation: str = "Sigmoid",
) -> torch.nn.Module:
    """Get the U-Net model for 2D segmentation tasks.

    Args:
        out_channels: The number of output channels of the network.
        initial_features: The number of features in the first level of the U-Net.
            The number of features increases by a factor of two in each level.
        final_activation: The activation applied to the last output layer.

    Returns:
        The U-Net.
    """
    model = UNet2d(
        in_channels=in_channels,
        out_channels=out_channels,
        initial_features=initial_features,
        gain=2,
        depth=4,
        final_activation=final_activation,
    )
    return model


def _adjust_patch_shape(data_shape, patch_shape):
    # If data is 2D and patch_shape is 3D, drop the extra dimension in patch_shape
    if data_shape == 2 and len(patch_shape) == 3:
        return patch_shape[1:]  # Remove the leading dimension in patch_shape
    return patch_shape  # Return the original patch_shape for 3D data


def _determine_ndim(patch_shape):
    # Check for 2D or 3D training
    try:
        z, y, x = patch_shape
    except ValueError:
        y, x = patch_shape
        z = 1
    is_2d = z == 1
    ndim = 2 if is_2d else 3
    return is_2d, ndim


def get_supervised_loader(
    data_paths: Tuple[str],
    raw_key: str,
    label_key: str,
    patch_shape: Tuple[int, int, int],
    batch_size: int,
    n_samples: Optional[int],
    add_boundary_transform: bool = True,
    label_dtype=torch.float32,
    rois: Optional[Tuple[Tuple[slice]]] = None,
    sampler: Optional[callable] = None,
    ignore_label: Optional[int] = None,
    label_transform: Optional[callable] = None,
    label_paths: Optional[Tuple[str]] = None,
    **loader_kwargs,
) -> torch.utils.data.DataLoader:
    """Get a dataloader for supervised segmentation training.

    Args:
        data_paths: The filepaths to the hdf5 files containing the training data.
        raw_key: The key that holds the raw data inside of the hdf5.
        label_key: The key that holds the labels inside of the hdf5.
        patch_shape: The patch shape used for a training example.
            In order to run 2d training pass a patch shape with a singleton in the z-axis,
            e.g. 'patch_shape = [1, 512, 512]'.
        batch_size: The batch size for training.
        n_samples: The number of samples per epoch. By default this will be estimated
            based on the patch_shape and size of the volumes used for training.
        add_boundary_transform: Whether to add a boundary channel to the training data.
        label_dtype: The datatype of the labels returned by the dataloader.
        rois: Optional region of interests for training.
        sampler: Optional sampler for selecting blocks for training.
            By default a minimum instance sampler will be used.
        ignore_label: Ignore label in the ground-truth. The areas marked by this label will be
            ignored in the loss computation. By default this option is not used.
        label_transform: Label transform that is applied to the segmentation to compute the targets.
            If no label transform is passed (the default) a boundary transform is used.
        label_paths: Optional paths containing the labels / annotations for training.
            If not given, the labels are expected to be contained in the `data_paths`.
        loader_kwargs: Additional keyword arguments for the dataloader.

    Returns:
        The PyTorch dataloader.
    """
    _, ndim = _determine_ndim(patch_shape)
    if label_transform is not None:  # A specific label transform was passed, do nothing.
        pass
    elif add_boundary_transform:
        if ignore_label is None:
            label_transform = torch_em.transform.BoundaryTransform(add_binary_target=True)
        else:
            label_transform = torch_em.transform.label.BoundaryTransformWithIgnoreLabel(
                add_binary_target=True, ignore_label=ignore_label
            )

    else:
        if ignore_label is not None:
            raise NotImplementedError
        label_transform = torch_em.transform.label.connected_components

    if ndim == 2:
        adjusted_patch_shape = _adjust_patch_shape(ndim, patch_shape)
        transform = torch_em.transform.Compose(
            torch_em.transform.PadIfNecessary(adjusted_patch_shape), torch_em.transform.get_augmentations(2)
        )
    else:
        transform = torch_em.transform.Compose(
            torch_em.transform.PadIfNecessary(patch_shape), torch_em.transform.get_augmentations(3)
        )

    num_workers = loader_kwargs.pop("num_workers", 4 * batch_size)
    shuffle = loader_kwargs.pop("shuffle", True)

    if sampler is None:
        sampler = torch_em.data.sampler.MinInstanceSampler(min_num_instances=4)

    if label_paths is None:
        label_paths = data_paths
    elif len(label_paths) != len(data_paths):
        raise ValueError(f"Data paths and label paths don't match: {len(data_paths)} != {len(label_paths)}")

    loader = torch_em.default_segmentation_loader(
        data_paths, raw_key,
        label_paths, label_key, sampler=sampler,
        batch_size=batch_size, patch_shape=patch_shape, ndim=ndim,
        is_seg_dataset=True, label_transform=label_transform, transform=transform,
        num_workers=num_workers, shuffle=shuffle, n_samples=n_samples,
        label_dtype=label_dtype, rois=rois, **loader_kwargs,
    )
    return loader


def supervised_training(
    name: str,
    train_paths: Tuple[str],
    val_paths: Tuple[str],
    label_key: str,
    patch_shape: Tuple[int, int, int],
    save_root: Optional[str] = None,
    raw_key: str = "raw",
    batch_size: int = 1,
    lr: float = 1e-4,
    n_iterations: int = int(1e5),
    train_label_paths: Optional[Tuple[str]] = None,
    val_label_paths: Optional[Tuple[str]] = None,
    train_rois: Optional[Tuple[Tuple[slice]]] = None,
    val_rois: Optional[Tuple[Tuple[slice]]] = None,
    sampler: Optional[callable] = None,
    n_samples_train: Optional[int] = None,
    n_samples_val: Optional[int] = None,
    check: bool = False,
    ignore_label: Optional[int] = None,
    label_transform: Optional[callable] = None,
    in_channels: int = 1,
    out_channels: int = 2,
    mask_channel: bool = False,
    device: int = 0,
    **loader_kwargs,
):
    """Run supervised segmentation training.

    This function trains a UNet for predicting outputs for segmentation.
    Expects instance labels and converts them to boundary targets.
    This behaviour can be changed by passing custom arguments for `label_transform`
    and/or `out_channels`.

    Args:
        name: The name for the checkpoint to be trained.
        train_paths: Filepaths to the hdf5 files for the training data.
        val_paths: Filepaths to the df5 files for the validation data.
        label_key: The key that holds the labels inside of the hdf5.
        patch_shape: The patch shape used for a training example.
            In order to run 2d training pass a patch shape with a singleton in the z-axis,
            e.g. 'patch_shape = [1, 512, 512]'.
        save_root: Folder where the checkpoint will be saved.
        raw_key: The key that holds the raw data inside of the hdf5.
        batch_size: The batch size for training.
        lr: The initial learning rate.
        n_iterations: The number of iterations to train for.
        train_label_paths: Optional paths containing the label data for training.
            If not given, the labels are expected to be part of `train_paths`.
        val_label_paths: Optional paths containing the label data for validation.
            If not given, the labels are expected to be part of `val_paths`.
        train_rois: Optional region of interests for training.
        val_rois: Optional region of interests for validation.
        sampler: Optional sampler for selecting blocks for training.
            By default a minimum instance sampler will be used.
        n_samples_train: The number of train samples per epoch. By default this will be estimated
            based on the patch_shape and size of the volumes used for training.
        n_samples_val: The number of val samples per epoch. By default this will be estimated
            based on the patch_shape and size of the volumes used for validation.
        check: Whether to check the training and validation loaders instead of running training.
        ignore_label: Ignore label in the ground-truth. The areas marked by this label will be
            ignored in the loss computation. By default this option is not used.
        label_transform: Label transform that is applied to the segmentation to compute the targets.
            If no label transform is passed (the default) a boundary transform is used.
        out_channels: The number of output channels of the UNet.
        mask_channel: Whether the last channels in the labels should be used for masking the loss.
            This can be used to implement more complex masking operations and is not compatible with `ignore_label`.
        device: GPU ID for training. 
        loader_kwargs: Additional keyword arguments for the dataloader.
    """
    train_loader = get_supervised_loader(train_paths, raw_key, label_key, patch_shape, batch_size,
                                         n_samples=n_samples_train, rois=train_rois, sampler=sampler,
                                         ignore_label=ignore_label, label_transform=label_transform,
                                         label_paths=train_label_paths, **loader_kwargs)
    val_loader = get_supervised_loader(val_paths, raw_key, label_key, patch_shape, batch_size,
                                       n_samples=n_samples_val, rois=val_rois, sampler=sampler,
                                       ignore_label=ignore_label, label_transform=label_transform,
                                       label_paths=val_label_paths, **loader_kwargs)

    if check:
        from torch_em.util.debug import check_loader
        check_loader(train_loader, n_samples=4)
        check_loader(val_loader, n_samples=4)
        return

    is_2d, _ = _determine_ndim(patch_shape)
    if is_2d:
        model = get_2d_model(out_channels=out_channels, in_channels=in_channels)
    else:
        model = get_3d_model(out_channels=out_channels, in_channels=in_channels)

    loss, metric = None, None
    # No ignore label -> we can use default loss.
    if ignore_label is None and not mask_channel:
        pass
    # If we have an ignore label the loss and metric have to be modified
    # so that the ignore mask is not used in the gradient calculation.
    elif ignore_label is not None:
        loss = torch_em.loss.LossWrapper(
            loss=torch_em.loss.DiceLoss(),
            transform=torch_em.loss.wrapper.MaskIgnoreLabel(
                ignore_label=ignore_label, masking_method="multiply",
            )
        )
        metric = loss
    elif mask_channel:
        loss = torch_em.loss.LossWrapper(
            loss=torch_em.loss.DiceLoss(),
            transform=torch_em.loss.wrapper.ApplyAndRemoveMask(
                masking_method="crop" if out_channels == 1 else "multiply")
        )
        metric = loss
    else:
        raise ValueError

    device = torch.device(f"cuda:{device}") if torch.cuda.is_available() else torch.device("cpu")
    trainer = torch_em.default_segmentation_trainer(
        name=name,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=lr,
        mixed_precision=True,
        log_image_interval=100,
        compile_model=False,
        save_root=save_root,
        loss=loss,
        metric=metric,
        device=device
    )
    trainer.fit(n_iterations)


def _derive_key_from_files(files, key):
    # Get all file extensions (general wild-cards may pick up files with multiple extensions).
    extensions = list(set([os.path.splitext(ff)[1] for ff in files]))

    # If we have more than 1 file extension we just use the key that was passed,
    # as it is unclear how to derive a consistent key.
    if len(extensions) > 1:
        return files, key

    ext = extensions[0]
    extension_to_key = {".tif": None, ".mrc": "data", ".rec": "data"}

    # Derive the key from the extension if the key is None.
    if key is None and ext in extension_to_key:
        key = extension_to_key[ext]
    # If the key is None and can't be derived raise an error.
    elif key is None and ext not in extension_to_key:
        raise ValueError(
            f"You have not passed a key for the data in {ext} format, for which the key cannot be derived."
        )
    # If the key was passed and doesn't match the extension raise an error.
    elif key is not None and ext in extension_to_key and key != extension_to_key[ext]:
        raise ValueError(
            f"The expected key {extension_to_key[ext]} for format {ext} did not match the passed key {key}."
        )
    return files, key


def _parse_input_folder(folder, pattern, key):
    files = sorted(glob(os.path.join(folder, "**", pattern), recursive=True))
    return _derive_key_from_files(files, key)


def _parse_input_files(args):
    train_image_paths, raw_key = _parse_input_folder(args.train_folder, args.image_file_pattern, args.raw_key)
    train_label_paths, label_key = _parse_input_folder(args.label_folder, args.label_file_pattern, args.label_key)
    if len(train_image_paths) != len(train_label_paths):
        raise ValueError(
            f"The image and label paths parsed from {args.train_folder} and {args.label_folder} don't match."
            f"The image folder contains {len(train_image_paths)}, the label folder contains {len(train_label_paths)}."
        )

    if args.val_folder is None:
        if args.val_label_folder is not None:
            raise ValueError("You have passed a val_label_folder, but not a val_folder.")
        train_image_paths, val_image_paths, train_label_paths, val_label_paths = train_test_split(
            train_image_paths, train_label_paths, test_size=args.val_fraction, random_state=42
        )
    else:
        if args.val_label_folder is None:
            raise ValueError("You have passed a val_folder, but not a val_label_folder.")
        val_image_paths = _parse_input_folder(args.val_image_folder, args.image_file_pattern, raw_key)
        val_label_paths = _parse_input_folder(args.val_label_folder, args.label_file_pattern, label_key)

    return train_image_paths, train_label_paths, val_image_paths, val_label_paths, raw_key, label_key


# TODO enable initialization with a pre-trained model.
def main():
    """@private
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Train a model for foreground and boundary segmentation via supervised learning.\n\n"
        "You can use this function to train a model for vesicle segmentation, or another segmentation task, like this:\n"  # noqa
        "synapse_net.run_supervised_training -n my_model -i /path/to/images -l /path/to/labels --patch_shape 32 192 192\n"  # noqa
        "The trained model will be saved in the folder 'checkpoints/my_model' (or whichever name you pass to the '-n' argument)."  # noqa
        "You can then use this model for segmentation with the SynapseNet GUI or CLI. "
        "Check out the information below for details on the arguments of this function.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("-n", "--name", required=True, help="The name of the model to be trained.")
    parser.add_argument("-p", "--patch_shape", nargs=3, type=int, help="The patch shape for training.")

    # Folders with training data, containing raw/image data and labels.
    parser.add_argument("-i", "--train_folder", required=True, help="The input folder with the training image data.")
    parser.add_argument("--image_file_pattern", default="*",
                        help="The pattern for selecting image files. For example, '*.mrc' to select all mrc files.")
    parser.add_argument("--raw_key",
                        help="The internal path for the raw data. If not given, will be determined based on the file extension.")  # noqa
    parser.add_argument("-l", "--label_folder", required=True, help="The input folder with the training labels.")
    parser.add_argument("--label_file_pattern", default="*",
                        help="The pattern for selecting label files. For example, '*.tif' to select all tif files.")
    parser.add_argument("--label_key",
                        help="The internal path for the label data. If not given, will be determined based on the file extension.")  # noqa

    # Optional folders with validation data. If not given the training data is split into train/val.
    parser.add_argument("--val_folder",
                        help="The input folder with the validation data. If not given the training data will be split for validation")  # noqa
    parser.add_argument("--val_label_folder",
                        help="The input folder with the validation labels. If not given the training data will be split for validation.")  # noqa

    # More optional argument:
    parser.add_argument("--batch_size", type=int, default=1, help="The batch size for training.")
    parser.add_argument("--n_samples_train", type=int, help="The number of samples per epoch for training. If not given will be derived from the data size.")  # noqa
    parser.add_argument("--n_samples_val", type=int, help="The number of samples per epoch for validation. If not given will be derived from the data size.")  # noqa
    parser.add_argument("--val_fraction", type=float, default=0.15, help="The fraction of the data to use for validation. This has no effect if 'val_folder' and 'val_label_folder' were passed.")  # noqa
    parser.add_argument("--check", action="store_true", help="Visualize samples from the data loaders to ensure correct data instead of running training.")  # noqa
    args = parser.parse_args()

    train_image_paths, train_label_paths, val_image_paths, val_label_paths, raw_key, label_key =\
        _parse_input_files(args)

    supervised_training(
        name=args.name, train_paths=train_image_paths, val_paths=val_image_paths,
        train_label_paths=train_label_paths, val_label_paths=val_label_paths,
        raw_key=raw_key, label_key=label_key, patch_shape=args.patch_shape, batch_size=args.batch_size,
        n_samples_train=args.n_samples_train, n_samples_val=args.n_samples_val,
        check=args.check,
    )
