import os
from typing import List
from clinicadl.utils.maps_manager.maps_manager import MapsManager
from logging import getLogger

logger = getLogger("clinicadl-pythae.pythae_train")

def train_pythae(maps_manager : MapsManager, split_list: List[int] = None):
    """
    Train using Pythae procedure
    only works for single splits
    """
    from pythae.pipelines import TrainingPipeline

    from clinithae.dataset.pythae_dataset import PythaeCAPS

    train_transforms, all_transforms = maps_manager.get_transforms(
        normalize=maps_manager.normalize,
        data_augmentation=maps_manager.data_augmentation,
        size_reduction=maps_manager.size_reduction,
        size_reduction_factor=maps_manager.size_reduction_factor,
    )

    split_manager = maps_manager._init_split_manager(split_list)
    for split in split_manager.split_iterator():
        logger.info(f"Training split {split}")

        model_dir = maps_manager.maps_path / f"split-{split}", "best-loss"
        if not model_dir.is_dir():
            model_dir.mkdir(parents=True)

        maps_manager.seed_everything(maps_manager.seed, maps_manager.deterministic, maps_manager.compensation)

        split_df_dict = split_manager[split]
        train_dataset = PythaeCAPS(
            maps_manager.caps_directory,
            split_df_dict["train"],
            maps_manager.preprocessing_dict,
            train_transformations=train_transforms,
            all_transformations=all_transforms,
        )
        eval_dataset = PythaeCAPS(
            maps_manager.caps_directory,
            split_df_dict["validation"],
            maps_manager.preprocessing_dict,
            train_transformations=train_transforms,
            all_transformations=all_transforms,
        )

        # Import the model
        clinicadl_model, _ = maps_manager._init_model(
            split=split,
            gpu=True,
        )
        model = clinicadl_model.model
        config = clinicadl_model.get_trainer_config(
            output_dir=model_dir,
            num_epochs=maps_manager.epochs,
            learning_rate=maps_manager.learning_rate,
            batch_size=maps_manager.batch_size,
        )
        # Create Pythae Training Pipeline
        pipeline = TrainingPipeline(training_config=config, model=model)
        # Create Pythae Training Pipeline
        pipeline = TrainingPipeline(training_config=config, model=model)

        # Launch training
        pipeline(
            train_data=train_dataset,  # must be torch.Tensor or np.array
            eval_data=eval_dataset,  # must be torch.Tensor or np.array
        )
        # Move saved model to the correct path in the MAPS
        src = model_dir / "*_training_*/final_model/model.pt"
        os.system(f"mv {src} {model_dir}")
