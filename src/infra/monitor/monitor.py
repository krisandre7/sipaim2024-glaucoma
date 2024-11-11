from typing import Any
import wandb
from src.infra.monitor.base import ExperimentMonitor


class WandbExperimentMonitor(ExperimentMonitor):

    def __init__(self, config: Any, dir = None,
                 entity: str = None, job_name: str = None, mode='online',
                 id=None, resume='allow', step=None, tags=None, project=None,
                 **kwargs):
        
        task = config.datamodule.args['task']
        group = 'justification' if task == 'multilabel' else 'referral'

        self.run = wandb.init(
            config=config,
            entity=entity,
            settings=wandb.Settings(job_name=job_name),
            group=group,
            tags=tags,
            mode=mode,
            id=id,
            resume=resume,
            dir=dir,
            project=project
        )
        self.log_table = wandb.Table(columns=["Image Name", "Output Image", "Ground Truth Image", "PSNR"])

    def log(self, args: Any):
        self.run.log(args)

    def watch_model(self, model, criterion, log="all", log_freq=10):
        self.run.watch(model, criterion, log=log, log_freq=log_freq)

    def log_image_comparison_table(self, image_name, output_image, ground_truth_image, psnr_score):
        self.log_table.add_data(image_name,
                                wandb.Image(output_image, caption="Output Image"),
                                wandb.Image(ground_truth_image, caption="Ground Truth Image"),
                                psnr_score)

    def finalize(self):
        # self.run.log({"Image Comparison Table": self.log_table})
        self.run.finish()
