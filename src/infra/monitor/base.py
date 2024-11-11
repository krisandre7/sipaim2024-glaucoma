from abc import ABC, abstractmethod


class ExperimentMonitor(ABC):

    @abstractmethod
    def watch_model(self, model, criterion, log="all", log_freq=10):
        """ Log informations about model in monitor"""

    @abstractmethod
    def log_image_comparison_table(self, image_name, output_image, ground_truth_image, psnr_score):
        """ Log image comparison table"""

    @abstractmethod
    def finalize(self):
        """ Add table to summary"""
